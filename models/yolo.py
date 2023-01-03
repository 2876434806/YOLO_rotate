# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device, time_sync

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):# 预测头
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        #self.angle = 180 # mean angle classifications CSL180 DCL具体选择
        self.angle = 8
        self.no = nc + 5 + self.angle  # number of outputs per anchor, 180 mean angle classifications.
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        '''
                m(
                    (0) :  nn.Conv2d(in_ch[0]（17）, (nc + 5 + self.angle) * na, kernel_size=1)  # 每个锚框中心点有3种尺度的anchor，每个anchor有 no 个输出
                    (1) :  nn.Conv2d(in_ch[1]（20）, (nc + 5 + self.angle) * na, kernel_size=1)
                    (2) :  nn.Conv2d(in_ch[2]（23）, (nc + 5 + self.angle) * na, kernel_size=1)
                )
        '''
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):    # nl=3
            x[i] = self.m[i](x[i])  # conv
            # x[i]:(batch_size, (5+nc+180) * na, size1', size2')
            # ny为featuremap的height， nx为featuremap的width
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)

            # x(batch_size,(5+nc+180) * 3,size1',size2') to x(batch_size,3种框,(5+nc+180),size1',size2')
            # x(batch_size,3种框,(5+nc+180),size1',size2') to x(batch_size, 3种框, size1', size2', (5+nc+180))
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                # grid[i].shape[2:4]=[size1, size2]  即[height/8*i, width/8*i] 与对应的featuremap层尺度一致
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    # grid[i]: tensor.shape(1, 1,当前featuremap的height, 当前featuremap的width, 2) 以height为y轴，width为x轴的grid坐标
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                # y:(batch_size,3种scale框,size1,size2,[xywh,score,num_classes,num_angle])
                y = x[i].sigmoid()
                # i : 0为small_forward 1为medium_forward 2为large_forward
                # self.grid[i]: tensor.shape(1, 1,当前featuremap的height, 当前featuremap的width, 2) 以height为y轴，width为x轴的grid坐标
                # grid坐标按顺序（0, 0） （1, 0）...  (width-1, 0) (0, 1) (1,1) ... (width-1, 1) ... (width-1 , height-1)
                # self.stride = ([ 8., 16., 32.])
                if self.inplace:
                    # xy 预测的真实坐标 y[..., 0:2] * 2. - 0.5是相对于左上角网格的偏移量； self.grid[i]是网格坐标索引
                    # 值域为（-0.5-1.5）
                    # yolov5可以跨半个网格点进行预测，提高了对格点周围的bbox的召回
                    # 解决了yolov3中因为sigmoid开区间而导致中心点无法到达边界处的问题
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    # anchor_grid[i].shape=(1, 3, 1, 1, 2)  y[..., 2:4].shape=(bs, 3, height', width', 2)
                    # wh 预测的真实wh  self.anchor_grid[i]是原始anchors宽高  (y[..., 2:4] * 2) ** 2 是预测出的anchors的wh倍率
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                # z:(batch_size, 累加3*size1*size2 , (5+nc+180)) z会一直在[1]维度上增添数据
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)
    # 所有的预测都是在gird层面，每一层下的grid的尺寸是不一样的
    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid

class ASFF_Detect(nn.Module):   #add ASFFV5 layer and Rfb
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter
    export = False

    def __init__(self, nc=2, anchors=(), multiplier=0.5,rfb=False,ch=(),inplace=True):  # detection layer
        super(ASFF_Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5 + 8 # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        # 与上面的detect层相比，不同的就是下面的所谓的fusion层
        self.l0_fusion = ASFFV5(level=0, multiplier=multiplier,rfb=rfb) #
        self.l1_fusion = ASFFV5(level=1, multiplier=multiplier,rfb=rfb)
        self.l2_fusion = ASFFV5(level=2, multiplier=multiplier,rfb=rfb)
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        result=[]
        # self.training |= self.export
        result.append(self.l2_fusion(x))    #l2层。result[0] : torch.Size([1, 128, 120, 120])
        result.append(self.l1_fusion(x))    # result[1] : torch.Size([1, 256, 60, 60])
        result.append(self.l0_fusion(x))    # result[2] : torch.Size([1, 512, 30, 30])
        # 经过上面的relust后，x的shape完全没变，变的只是里面的值。即经过了ASFF特征融合了，后面的代码和前面的detect中的完全一样
        x=result
        for i in range(self.nl):    # nl  三种步长的检测网络
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Decoupled_Detect1(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()

        self.nc = nc  # number of classes
        self.no = nc + 5 + 8  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(DecoupledHead1(x, nc, anchors) for x in ch)
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(y, x, indexing='ij')
        else:
            yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid

class Decoupled_Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.angle = 8
        self.no = nc + 5 + self.angle  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        # self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.m_box = nn.ModuleList(nn.Conv2d(256, 4 * self.na, 1) for x in ch)  # output conv
        self.m_conf = nn.ModuleList(nn.Conv2d(256, 1 * self.na, 1) for x in ch)  # output conv
        self.m_labels = nn.ModuleList(nn.Conv2d(256, self.nc * self.na, 1) for x in ch)  # output conv
        self.m_angles = nn.ModuleList(nn.Conv2d(256, self.angle * self.na, 1) for x in ch)  # output conv
        self.base_conv = nn.ModuleList(BaseConv(in_channels=x, out_channels=256, ksize=1, stride=1) for x in ch)
        self.cls_convs = nn.ModuleList(BaseConv(in_channels=256, out_channels=256, ksize=3, stride=1) for x in ch)
        self.reg_convs = nn.ModuleList(BaseConv(in_channels=256, out_channels=256, ksize=3, stride=1) for x in ch)

        # self.m = nn.ModuleList(nn.Conv2d(x, 4 * self.na, 1) for x in ch, nn.Conv2d(x, 1 * self.na, 1) for x in ch,nn.Conv2d(x, self.nc * self.na, 1) for x in ch)
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)self.ch = ch

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            # # x[i] = self.m[i](x[i])  # convs
            # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&", i)
            # print(x[i].shape)
            # print(self.base_conv[i])
            # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

            x_feature = self.base_conv[i](x[i])
            # x_feature = x[i]

            cls_feature = self.cls_convs[i](x_feature)
            reg_feature = self.reg_convs[i](x_feature)
            # reg_feature = x_feature

            m_box = self.m_box[i](reg_feature)
            m_conf = self.m_conf[i](reg_feature)
            m_labels = self.m_labels[i](cls_feature)
            m_angles = self.m_angles[i](cls_feature)
            x[i] = torch.cat((m_box, m_conf, m_labels,m_angles), 1)
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module



class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        # print(self.bn(self.conv(x)).shape)
        return self.act(self.bn(self.conv(x)))
        # return self.bn(self.conv(x))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict  有预训练权重文件时cfg加载权重中保存的cfg字典内容；
        else:  # is *.yaml   没有预训练权重文件时加载用户定义的opt.cfg权重文件路径，再载入文件中的内容到字典中
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels # 确定输入channel
        if nc and nc != self.yaml['nc']:  # input channels 字典中的nc与data.yaml中的nc不同，则以data.yaml中的nc为准
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)
        # 调用一次forward()函数，输入了一个[1, C, 256, 256]的tensor，然后得到FPN输出结果的维度。
        # 然后求出了下采样的倍数stride：8，16，32。
        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect) or isinstance(m,Decoupled_Detect) or (m, ASFF_Detect) or isinstance(m,Decoupled_Detect1):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            # 将anchor放缩到了3个不同的尺度上
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            # self._initialize_biases()  # only run once
            try :
                self._initialize_biases()  # only run once
                LOGGER.info('initialize_biases done')
            except :
                LOGGER.info('decoupled no biase ')

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales//
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        """
                该函数为前向计算函数，输入向量经函数计算后，返回backbone+head+detect计算结果
                @param x: 待前向传播的向量 size=(batch_size, 3, height, width)
        """
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer即if current layer is concat or spp
                # 例子：m=Concat层函数 m.f = [-1, 4], x = [x,y[4]] ,即x= [上一层的前向计算结果, 第四层的前向计算结果]
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect) or isinstance(m,Decoupled_Detect) or (m, ASFF_Detect) or isinstance(m,Decoupled_Detect1) # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect)or isinstance(m,Decoupled_Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


def parse_model(d, ch):  # model_dict, input_channels(3)
    # 读出配置dict里面的参数
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    # na是判断anchor的数量
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    # no = na * (nc + 180 + 5)  # number of outputs = anchors * (classes + 5)
    # no是根据anchor数量推断的输出维度
    no = na * (nc + 8 + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    '''
        从yaml文件中读取模型网络结构参数
        from : -1 代表是从上一层获得的输入;    -2表示从上两层获得的输入（head同理）
        number : module重复的次数
        module : 功能模块 common.py中定义的函数
        args : 功能函数的输入参数定义
    '''
    # 开始迭代循环backbone与head的配置
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # isinstance(object, classinfo)
        # 如果参数object是classinfo的实例，或者object是classinfo类的子类的一个实例， 返回True。如果object不是一个给定类型的的对象， 则返回结果总是False。
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                # 若arg参数为字符串，则直接执行表达式(如Flase None等)，否则直接等于数字本身（如64，128等）
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass
        # 网络用n*gd控制模块的深度缩放
        # 模型重复次数为1时，n为1，否则n = （n*gd)向上取整
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 CoordAtt, BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, C3STR, CBAM, CARAFE, CTR3, C3SE, C3CA, C3ECA,FSM,CSP,max_pool,imp_SPPF,C3Ghost_ca,C3Ghost_eca,C3Ghost_cbam]:
            c1, c2 = ch[f], args[0]  # c1 第一次等于3，之后为ch[-1]代表着上一个模块的输出通道，c2 = 每次module函数中的out_channels参数
            # 此时参数的输出层厚度不等于no
            # 即如果不是最后的输出层
            if c2 != no:  # if not output
                # 配合make_divisible()函数，是为了放缩网络模块的宽度（既输出的通道数）
                # make_divisible()函数保证了输出的通道是8的倍数。
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            # 只有BottleneckCSP、C3、C3TR和C3Ghost这些module会根据深度参数n被调整该模块的重复迭加次数。
            if m in [BottleneckCSP, C3, C3TR, C3STR, C3Ghost,CTR3,C3SE,C3CA,C3ECA,C3Ghost_ca,C3Ghost_eca,C3Ghost_cbam]:
                args.insert(2, n)  # number of repeats
                n = 1
            # args保存的前两个参数为moudle的输入通道数和输出通道数
        elif m in [ VoVGSCSP, GSConv]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [ VoVGSCSP]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m in [SKAttention]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, *args[1:]]
        elif m in [mutil_scale_SKAttention_add, mutil_scale_ESKAttention_add,mutil_scale_ESKAttention_add1,mutil_scale_SKAttention_add1,mutil_scale_CASKAttention_add]:
            c2 = max(ch[x] for x in f)
            args = [c2]
        elif m in [mutil_scale_SKAttention_concat,mutil_scale_ESKAttention_concat,mutil_scale_ESKAttention_concat1,mutil_scale_SKAttention_concat1,mutil_scale_CASKAttention_concat]:
            c2 = sum(ch[x] for x in f)
            args = [c2]
        elif m in [CoT3]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [CoT3]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        # Concat，f是所有需要拼接层的index，则输出通道c2是所有层的和。
        # 以5s文件里第一个Concat为例，ch[-1] + ch[x + 1] = ch[-1] + ch[7] = 640 + 640 = 1280
        elif m in [Concat, BiFPN_Concat2, BiFPN_Concat3]:
            c2 = sum(ch[x] for x in f)
        elif m in [BiFPN_Add2, BiFPN_Add3,FAM]:
            c2 = max([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Decoupled_Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)

        elif m is Decoupled_Detect1:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)

        elif m is ASFF_Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        '''以第一层focus为例
                args： [ch[-1], out_channels, kernel_size, strides(可能)] = [3, 80, 3]
                m: class 'models.common.Focus'
                m_: Focus(  # focus函数会在一开始将3通道的图像再次分为12通道
                         (conv): Conv(
                                      (conv): Conv2d(12, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                                      (bn): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                      (act): Hardswish()
                                      )
                          )
        '''
        # args用于构建module m 模块循环次数通过n来确定
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module

        # 这里做了一些输出打印，可以看到每一层module构建的编号、参数量等情况
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        # 构建的模块保存到layers里，把该层的输出通道数写入ch列表里。
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='C:\Code\yolov5_rotation-master\models\\remodel2\yolov5m.yaml', help='model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(FILE.stem, opt)
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Test all models
    if opt.test:
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
