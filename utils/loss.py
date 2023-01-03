# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
import numpy as np
import math

from utils.general import xywh2xyxy
from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel
from utils.angledclcode import angle_label_encode, angle_label_decode
from utils import *


#标签平滑
def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class FocalLoss_A(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true, wadaarsw):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor * wadaarsw

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def gaussian_label(label, num_class, u=0, sig=4.0):
    '''
    转换成CSL Labels：
        用高斯窗口函数根据角度θ的周期性赋予gt labels同样的周期性，使得损失函数在计算边界处时可以做到“差值很大但loss很小”；
        并且使得其labels具有环形特征，能够反映各个θ之间的角度距离
    @param label: 当前box的θ类别  shape(1)
    @param num_class: θ类别数量=180
    @param u: 高斯函数中的μ
    @param sig: 高斯函数中的σ
    @return: 高斯离散数组:将高斯函数的最高值设置在θ所在的位置，例如label为45，则将高斯分布数列向右移动直至x轴为45时，取值为1 shape(180)
    '''
    # floor()返回数字的下舍整数   ceil() 函数返回数字的上入整数  range(-90,90)
    # 以num_class=180为例，生成从-90到89的数字整形list  shape(180)
    x = np.array(range(math.floor(-num_class / 2), math.ceil(num_class / 2), 1))
    y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2))  # shape(180) 为-90到89的经高斯公式计算后的数值
    # 将高斯函数的最高值设置在θ所在的位置，例如label为45，则将高斯分布数列向右移动直至x轴为45时，取值为1
    return np.concatenate([y_sig[math.ceil(num_class / 2) - int(label.item()):],
                           y_sig[:math.ceil(num_class / 2) - int(label.item())]], axis=0)




class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters
        self.class_index = 5 + model.nc
        self.threshold = 0.5 # 将预测值解码时，即二进制转为十进制，获得二进制编码的门限值
        self.aspect_ratio_threshold = 1.5 # 纵横比加权因子的门限值
        self.omega = 180 / 256.

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        BCEangle = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['angle_pw']])).to(device)

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 2.0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
            BCEangle = FocalLoss_A(BCEangle, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        # 设置三个特征图对应输出的损失系数  4.0, 1.0, 0.4分别对应下采样8, 16, 32
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.BCEangle = BCEangle
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        langle = torch.zeros(1, device=device)
        #获取标签分类，边框，索引，anchors
        tcls, tangle, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            # 根据indices获取索引，方便找到网格输出
            # pi.shape = torch.size([])
            # pi.size = (batch_size, 3种scale框, size1, size2, [xywh, score, num_classes, num_angles])
            # indice[i] = (该image属于该batch的第几个图片, 该box属于哪种scale的anchor, 网格索引1, 网格索引2)
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            # tobj.size = (batch_size, 3种scale框, feature_height, feature_width, 1) 全为0
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets 预测框的个数
            if n:
                # 前向传播的结果和target信息进行匹配 筛选对应的网格 得到对应网格的前向传播结果
                # pi : 前向传播结果  b, a, gj, gi : target信息
                # 对应网格的前向传播结果 ps.shape = torch.Size([312, 15]) 312表示312个目标
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                # 对前向传播结果xy进行回归  （预测的是offset）-> 处理成与当前网格左上角坐标的xy偏移量
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                # 对前向传播结果wh进行回归  （预测的是当前featuremap尺度上的框wh尺度缩放量）-> 处理成featuremap尺度上框的真实wh
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # 计算边框损失，注意这个CIoU=True，计算的是ciou损失
                # 3个tensor组成的list (box[i])  对每个步长网络生成对应的gt_box tensor
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, SIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness 原yolov5的Objectness损失，即完全解耦预测角度与预测置信度之间的关联，置信度只与边框参数有关联
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:self.class_index], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:self.class_index], t)  # BCE
                
                # # Θ类别损失（CSL）
                # ttheta = torch.zeros_like(ps[:, self.class_index:])  # size(num, 180)
                # for idx in range(len(ps)):  # idx start from 0 to len(ps)-1
                #     # 3个tensor组成的list (tensor_angle_list[i])  对每个步长网络生成对应的class tensor  tangle[i].shape=(num_i, 1)
                #     theta = tangle[i][idx]  # 取出第i个layer中的第idx个目标的角度数值  例如取值θ=90
                #     # CSL论文中窗口半径为6效果最佳，过小无法学到角度信息，过大则角度预测偏差加大
                #     csl_label = gaussian_label(theta, 180, u=0, sig=12)  # 用长度为1的θ值构建长度为180的csl_label
                #     ttheta[idx] = torch.from_numpy(csl_label)  # 将cls_label放入对应的目标中
                # langle += self.BCEangle(ps[:, self.class_index:], ttheta)


                # Θ类别损失（DCL）

                device = torch.device('cuda:0')
                theta = tangle[i]
                theta_numpy = theta.cpu().numpy() # gt 10
                # 计算标签的二进制值 用于计算损失
                # gt_pro = (- round(theta_numpy - 90)/ self.omega)
                gt_binary = angle_label_encode(theta_numpy, 180, omega=self.omega, mode=1) # gt 10 --> 2
                # 获取预测的十进制值 用于计算Wadarsw

                pre_origin = torch.sigmoid(ps[:, self.class_index:])
                pre_origin = pre_origin.cpu().detach().numpy()# 获得经过sigmoid函数后的原始预测值
                pre_origin = np.where(pre_origin < self.threshold, 0, 1) # 根据门限值将预测值改为二进制码0或1
                # pre_origin_pro = 90 - int(round(sigmoid))
                pre_decimal = angle_label_decode(pre_origin, 180, omega=self.omega, mode=1)# pre 2 -> 10
                # 获取gt纵横比 用于计算Wadarsw
                box = tbox[i]
                ratio = box[:,2:3]/box[:,3:4]
                # 获得Wadarsw 解决类正方体问题以及对于角度预测的容忍性问题
                alpha = torch.where(ratio > self.aspect_ratio_threshold, 1, 2) #
                pre_decimal = torch.tensor(pre_decimal)
                gt_binary = torch.tensor(gt_binary)
                pre_decimal = pre_decimal.reshape(-1,1)
                theta = theta.reshape(-1,1)
                wadaarsw = alpha * (theta - pre_decimal.to(device))
                wadaarsw = wadaarsw.sin().abs()
                # 计算角度损失
                langle_this_batch = self.BCEangle(ps[:, self.class_index:], gt_binary.to(device)) * wadaarsw
                #langle_this_batch = self.BCEangle(ps[:, self.class_index:], gt_binary.to(device))
                langle += sum(langle_this_batch)/langle_this_batch.shape[0]

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        langle *= self.hyp['angle']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls + langle) * bs, torch.cat((lbox, lobj, lcls, langle)).detach()

    def build_targets(self, p, targets):# 用于生成网络训练时所需要的目标框 即生成正样本，正样本数量大于真实的目标数量
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        # na为锚框的种类数 nt为目标数
        na, nt = self.na, targets.shape[0]  # number of anchors, targets

        tcls, tbox, indices, anch = [], [], [], []
        tangle = []
        # 利用gain来计算目标在某一个特征图上的位置信息，初始化为1
        gain = torch.ones(8, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # 添加anchor索引，将每一个目标分配给3个anchor
        # ai：[3,10] = >
        #targets 格式转变由 [10, 7] => [3,10,8]
        # targets shape为[3,10,8] =>[na, nt, (哪张图片的目标1+类别1+位置4+anchor索引)] //相当于给每个目标都先分配三个框
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl): # number of layers
            anchors = self.anchors[i] #self.anchor是相对于当前特征层的宽高值
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            # Match targets to anchors # 将targets 的xywh转化为相对于当前特征曾的值
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio 取出t的wh值，分别和当前层的3个anchor进行比较
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter 对之前每个目标三个分配的anchor进行筛选，即将没有与当前层anchor匹配成功的t删除掉

                # Offsets 选取周边格子作为预测 也相当于用周边格子预测该物体从而增加正样本数量
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T # j控制左边网格，k控制上面网格
                l, m = ((gxi % 1 < g) & (gxi > 1)).T # l控制右边网格，m控制下边网格
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            angle = t[:, 6].long()  # angle
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices
            #gi = torch.clamp(gi, 0, num_w - 1)
            #gj = torch.clamp(gi, 0, num_h - 1)

            # Append
            a = t[:, 7].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            tangle.append(angle)  # angle

        return tcls, tangle, tbox, indices, anch





