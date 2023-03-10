import torch
import torch.nn as nn
from torch.nn.functional import fold, unfold
from torch import empty

## https://johnwlambert.github.io/conv-backprop/

def convolution(input,weight,padding=(0,0),stride=(1,1),dilation=(1,1)):
        output_H = ((input.size(2)+2*padding[0]-dilation[0]*(weight.size(2)-1)-1)/stride[0])+1
        output_W = ((input.size(3)+2*padding[0]-dilation[0]*(weight.size(3)-1)-1)/stride[0])+1
        output_C = weight.size(0)
        unfolded = unfold(input, kernel_size=(weight.size(2),weight.size(3)),padding=padding,stride=stride, dilation=dilation)
        #print(unfolded.size())
        product =  unfolded.transpose(1, 2).matmul(weight.view(output_C, -1).t()).transpose(1, 2)
        #print(product.size())
        folded = fold(product, (int(output_H), int(output_W)), kernel_size=(1, 1))
        #print(folded.size())
        return folded

def zero_padding(input, pad_size=(1,1)):
        output = empty((input.size(0), input.size(1), input.size(2)+(pad_size[0]*2), input.size(3)+(pad_size[1]*2))).fill_(0.)
        output[:,:,pad_size[0]:-(pad_size[0]),pad_size[1]:-(pad_size[1])]=input
        return output

def dilate(input,dil_size):
        output_H = input.size(2)+(input.size(3)-1)*(dil_size-1)
        output_W = input.size(2)+(input.size(3)-1)*(dil_size-1)
        output = empty((input.size(0), input.size(1), output_H, output_W)).fill_(0.)
        output[:,:,::dil_size,::dil_size] = input
        return output


overall_padding = 1
overall_stride = 2

"""x = torch.tensor([
        [1,1,1,2,3,4,5,6],
        [1,1,1,2,3,4,5,6],
        [1,1,1,2,3,4,5,6],
        [2,2,2,2,3,4,5,6],
        [3,3,3,3,3,4,5,6],
        [4,4,4,4,4,4,5,6],
        [5,5,5,5,5,5,5,6],
        [6,6,6,6,6,6,6,6]
    ]).to(torch.float32).view(1,1,8,8)"""

x = torch.randint(0,10,(5,3,5,5)).to(torch.float32)

"""x = torch.tensor([
        [1,1,1,2,3],
        [1,1,1,2,3],
        [1,1,1,2,3],
        [2,2,2,2,3],
        [3,3,3,3,3],
    ]).to(torch.float32).view(1,1,5,5)"""

"""x = torch.tensor([
        [1,1,1,2],
        [1,1,1,2],
        [1,1,1,2],
        [2,2,2,2]
    ]).to(torch.float32).view(1,1,4,4)"""

x.requires_grad = True

"""k = torch.tensor([
        [1,0,-1],
        [2,0,-2],
        [1,0,-1]
    ]).to(torch.float32).view(1,1,3,3)"""

"""k = torch.tensor([
        [1,0],
        [2,0]
    ]).to(torch.float32).view(1,1,2,2)"""
k = torch.randint(0,10,(2,3,3,3)).to(torch.float32)

conv = torch.nn.Conv2d(
    in_channels=k.size(1),
    out_channels=k.size(0),
    kernel_size=k.size(3),
    bias=False,
    stride = overall_stride,
    padding_mode='zeros',
    padding=overall_padding
)

conv.weight = torch.nn.Parameter(k)
out = conv(x)
loss = out.sum()
loss.backward()

"""print("using pytorch")
print("out")
print(out)
print("w grad")
print(conv.weight.grad)
print("x grad")
print(x.grad)
"""

## handmade forward and backward conv
"""
print("###################")
print("manually")"""
fol = convolution(x,k,padding=(overall_padding,overall_padding),stride=(overall_stride,overall_stride))
torch.testing.assert_allclose(out , fol)
print("passed test 1")
"""print("out")
print(fol)"""
## upstrem gradient (dLoss/dOutput)

"""
ups_grad = torch.empty(out.size()).fill_(1.)
## x * ups_grad convolution
print(x.size())
print(ups_grad.size())
w_grad = convolution(x,ups_grad,padding=(overall_padding,overall_padding),dilation=(overall_stride,overall_stride))

print("w grad")
print(w_grad[:,:,:k.size(2),:k.size(3)])
w_grad = w_grad[:,:,:k.size(2),:k.size(3)]
## 
#padded = zero_padding(ups_grad,pad_size=(1,1))
#print(padded)
dilated_ups_grad = dilate(ups_grad,overall_stride)
x_grad = empty(x.size()).fill_(0.)
out = convolution(dilated_ups_grad, k.rot90(2,(2,3)), padding=(k.size(3)-overall_padding-1,k.size(2)-overall_padding-1))
x_grad[:out.size(0),:out.size(1),:out.size(2),:out.size(3)]=out
print("x grad")
print(x.grad)
print(out)

torch.testing.assert_allclose(conv.weight.grad , w_grad)
print("passed test 2")
torch.testing.assert_allclose(x.grad , x_grad)
print("passed test 3")
"""
