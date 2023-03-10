import torch
from torch import empty, arange, cat



class Sigmoid (object) :
	def __init__(self):
		"""ReLU module declaration"""
	def forward (self, *input) :
		#self.output = empty(input[0].size()).copy_(input[0])
		self.input = empty(input[0].size()).copy_(input[0])
		self.output = self.input.mul(-1).exp().add(1)**(-1)
		return self.output
	def backward (self, *gradwrtoutput):
		self.grad_x = self.input.mul(-1).exp().mul(self.input.mul(-1).exp().add(1)**(-2))
		return self.grad_x.mul(gradwrtoutput[0])
	def param (self) :
		return []

x = empty((10,3,4,4)).normal_(0,1)
x.requires_grad = True
print(x)

torch_sigmoid = torch.nn.Sigmoid()
my_sigmoid = Sigmoid()


torch_out = torch_sigmoid(x)
loss = torch_out.sum()
loss.backward()
torch_grad = x.grad.mul(torch_out)
my_out = my_sigmoid.forward(x)

torch.testing.assert_allclose(my_out , torch_out)
print("same output")

my_grad = my_sigmoid.backward(my_out)

print(torch_grad)
print(my_grad)
torch.testing.assert_allclose(my_grad , torch_grad)
print("same gradient")