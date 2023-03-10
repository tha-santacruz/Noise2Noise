import torch
from torch import empty, arange, cat
from torch.nn.functional import fold, unfold

# flat =  x.view(x.size(0)*x.size(1),x.size(2)*x.size(3))
# repeated = flat.unsqueeze(2).repeat(1,1,scale**2)
# folded = fold(repeated.transpose(1,2),output_size=(x.size(2)*scale,x.size(3)*scale),kernel_size=scale,stride=scale)
# upsampled = folded.view(x.size(0),x.size(1),folded.size(2),folded.size(3))

class Upsampling (object) :
	def __init__(self, scale_factor):
		"""Upsampling module declaration
		This module contains a 2D nearest neighbor upsampling algorithm
		The scale attribute stores the ration of output to input sizes"""
		self.scale = scale_factor
	def forward (self, *input) :
		self.input = empty(input[0].size()).copy_(input[0])
		output_size = (
			self.input.size(0),
			self.input.size(1),
			self.input.size(2)*self.scale,
			self.input.size(3)*self.scale,)
		flat =  self.input.view(self.input.size(0)*self.input.size(1),self.input.size(2)*self.input.size(3))
		repeated = flat.unsqueeze(2).repeat(1,1,self.scale**2)
		folded = fold(repeated.transpose(1,2),output_size=(self.input.size(2)*self.scale,self.input.size(3)*self.scale),kernel_size=self.scale,stride=self.scale)
		self.output = folded.view(x.size(0),x.size(1),folded.size(2),folded.size(3))
		return self.output
	def backward (self, *gradwrtoutput):
		unfolded = unfold(gradwrtoutput[0], kernel_size=self.scale, stride=self.scale)
		self.grad_x = unfolded.view(self.input.size(0),self.input.size(1),self.scale**2,self.input.size(2),self.input.size(3)).sum(dim=2)
		return self.grad_x
	def param (self) :
		return []

x = torch.randint(0,10,(2,3,2,2)).to(torch.float32)
x.requires_grad = True
print(x)

torch_upsampling = torch.nn.UpsamplingNearest2d(scale_factor=3)
my_upsampling = Upsampling(scale_factor=3)


torch_out = torch_upsampling(x)
loss = torch_out.sum()
loss.backward()
dloss = empty(torch_out.size()).fill_(1.)
torch_grad = x.grad
my_out = my_upsampling.forward(x)

torch.testing.assert_allclose(my_out , torch_out)
print("same output")

my_grad = my_upsampling.backward(dloss)

print(torch_grad.size())
print(my_grad.size())
torch.testing.assert_allclose(my_grad , torch_grad)
print("same gradient")





