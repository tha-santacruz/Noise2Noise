import torch
import torch.nn as nn
from torch.nn.functional import fold, unfold
from torch import empty

overall_stride=1
overall_padding=0

x = torch.randint(0,10,(2,2,3,3)).to(torch.float32)
x.requires_grad = True
k = torch.randint(0,10,(3,2,2,2)).to(torch.float32)
b = torch.randint(0,10,(k.size(0),)).to(torch.float32)
print(x)
print(k)
print(b)

torch_conv = torch.nn.Conv2d(
    in_channels=2,
    out_channels=3,
    kernel_size=3,
    bias=True,
    stride = overall_stride,
    padding_mode='zeros',
    padding=overall_padding
)

torch_conv.weight = torch.nn.Parameter(k)
torch_conv.bias = torch.nn.Parameter(b)

torch_out = torch_conv(x)
loss = torch_out.sum()
loss.backward()

print("pytorch :")
print(f"x grad : {x.grad}")
print(f"w grad : {torch_conv.weight.grad}")
print(f"b grad : {torch_conv.bias.grad}")