import torch
from torch import empty, arange, cat



class ReLU (object) :
	def __init__(self):
		"""ReLU module declaration"""
	def forward (self, *input) :
		self.output = empty(input[0].size()).copy_(input[0])
		self.map = self.output<0
		self.output[self.map]=0.
		return self.output
	def backward (self, *gradwrtoutput):
		self.grad_x = empty(self.output.size()).fill_(1.)
		self.grad_x[self.map] = 0.
		return self.grad_x.mul(gradwrtoutput[0])
	def param (self) :
		return []

x = empty((10,3,4,4)).normal_(0,1)
x.requires_grad = True
print(x)

torch_relu = torch.nn.ReLU()
my_relu = ReLU()


torch_out = torch_relu(x)
loss = torch_out.sum()
loss.backward()
torch_grad = x.grad.mul(torch_out)
my_out = my_relu.forward(x)

torch.testing.assert_allclose(my_out , torch_out)
print("same output")

my_grad = my_relu.backward(my_out)

print(torch_grad)
print(my_grad)
torch.testing.assert_allclose(my_grad , torch_grad)
print("same gradient")