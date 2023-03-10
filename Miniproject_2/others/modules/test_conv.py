import torch
from torch import empty, arange, cat
from torch.nn.functional import unfold, fold

class Parameter (object):
    def __init__(self, value):
        self.value = value
    def __get__(self, instance, owner):
        return self.value
    def set_value(self,new_value):
        self.value = new_value
    def add_value(self, new_value):
        self.value = self.value+new_value
    def zero_value(self):
        self.value = self.value.fill_(0.)

class Conv2d (object) :
    def __init__(self, in_channels, out_channels, kernel_size=(2,2), padding=0, stride=1, dilation=1):
        """Conv2d module declaration"""
        ## arguments as attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

        ## weights initialization : from pytorch documentation
        ## see https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.k = 1/(self.in_channels*self.kernel_size[0]*self.kernel_size[1])
        self.weight = Parameter(empty(self.out_channels,self.in_channels, self.kernel_size[0], self.kernel_size[1]).uniform_(-(self.k**0.5), self.k**0.5))
        self.bias = Parameter(empty(self.out_channels).uniform_(-self.k, self.k))

        ## gradient tensors initialization
        self.weight_grad = Parameter(empty(self.out_channels,self.in_channels, self.kernel_size[0], self.kernel_size[1]).fill_(0.))
        self.bias_grad = Parameter(empty(self.out_channels).uniform_(-self.k, self.k).fill_(0.))

    def forward (self, *input) :
        ## convolution : inspired by the pytorch documentation
        ## see https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
        self.input = empty(input[0].size()).copy_(input[0])
        output_H = ((self.input.size(2)+2*self.padding-self.dilation*(self.weight.value.size(2)-1)-1)/self.stride)+1
        output_W = ((self.input.size(3)+2*self.padding-self.dilation*(self.weight.value.size(3)-1)-1)/self.stride)+1
        self.unfolded = unfold(self.input, 
            kernel_size=(self.weight.value.size(2),self.weight.value.size(3)),
            padding=self.padding,
            stride=self.stride, 
            dilation=self.dilation)
        product =  self.unfolded.transpose(1, 2).matmul(self.weight.value.view(self.out_channels, -1).t()).transpose(1, 2) + self.bias.value.view(1, -1, 1)
        self.output = fold(product, (int(output_H), int(output_W)), kernel_size=(1, 1))
        return self.output
    def backward (self, *gradwrtoutput):
    	## bias gradient
        self.gradwrtoutput = gradwrtoutput[0]
        self.bias_grad.add_value(self.gradwrtoutput.sum(dim=(0,2,3)))

        ## weight gradient
        temp1 = self.gradwrtoutput.view(self.gradwrtoutput.size(0),self.gradwrtoutput.size(1),-1)
        temp2 = self.unfolded.transpose(1,2)
        product1 = temp1.matmul(temp2)
        self.weight_grad.add_value(product1.sum(dim=0).view(self.weight_grad.value.size()))

        ## input gradient
        temp3 = self.weight.value.view(self.out_channels, -1).t()
        product2 = temp3.matmul(temp1)
        x_grad = fold(product2, 
            (self.input.size(2),self.input.size(3)),
            kernel_size=(self.weight.value.size(2),self.weight.value.size(3)),
            padding=self.padding,
            stride=self.stride, 
            dilation=self.dilation)
        return x_grad

        pass
    def param (self) :
        return [[self.weight, self.weight_grad], [self.bias, self.bias_grad]]


overall_stride = 2
overall_padding = 1

x = empty((10,3,5,5)).normal_(0,1)
x.requires_grad = True
print(x)

k = empty((2,3,3,3)).normal_(0,1)
print(k)

b = empty(k.size(0)).normal_(0,1)
print(b)

torch_conv = torch.nn.Conv2d(
    in_channels=k.size(1),
    out_channels=k.size(0),
    kernel_size=k.size(3),
    bias=True,
    stride = overall_stride,
    padding_mode='zeros',
    padding=overall_padding
)


torch_conv.weight = torch.nn.Parameter(k)
torch_conv.bias = torch.nn.Parameter(b)

my_conv = Conv2d(
    in_channels=k.size(1),
    out_channels=k.size(0),
    kernel_size=(k.size(2),k.size(3)),
    stride = overall_stride,
    padding=overall_padding,
)

my_conv.weight.set_value(k)
my_conv.bias.set_value(b)

torch_out = torch_conv(x)
loss = torch_out.sum()
loss.backward()

my_out = my_conv.forward(x)
torch.testing.assert_allclose(torch_out , my_out)
print("passed test 1")

my_xgrad = my_conv.backward(empty(my_out.size()).fill_(1.))

torch.testing.assert_allclose(torch_conv.bias.grad , my_conv.bias_grad.value)
print("passed test 2")

torch.testing.assert_allclose(torch_conv.weight.grad , my_conv.weight_grad.value)
print("passed test 3")

torch.testing.assert_allclose(x.grad , my_xgrad)
print("passed test 4")


"""torch_xgrad = x.grad
print("pytorch :")
print(f"x grad : {torch_xgrad}")
print(f"w grad : {torch_conv.weight.grad}")
print(f"b grad : {torch_conv.bias.grad}")"""

print(my_conv.weight)