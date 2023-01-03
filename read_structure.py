'查看模型信息'
import torch

model = "F:\Code\部分训练结果存档\\6.22日之后\exp69\weights\\best.pt"

net = torch.load(model)
print(net["model"])