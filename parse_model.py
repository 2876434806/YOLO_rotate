'估计参数数量'
import torch
from thop import profile
import pandas as pd
import torchvision

Retinanet = "F:\Code\yolov5-master\Retinanet_res50_800_3a.pth"
yolo_5m = "F:\Code\yolov5_rotation-master\yolov5m.pt"

net = torch.load(Retinanet)
print(type(net)) # 类型是 dict
print(len(net)) # 长度为 4，即存在四个 key-value 键值对
for k in net.keys():
    print(k) # 查看四个键，分别是 model,optimizer,scheduler,iteration
print(type(net["model"]))
sum = 0

for key,value in net["model"].items():
    param = value.size()
    length = len(value.size())
    num = 1
    if length != 0 :
        for i in range(length) :
            num = num * param[i]
    sum = sum + num
    print(key,value.size(), num, sep=" ")

print("sum" , sum)