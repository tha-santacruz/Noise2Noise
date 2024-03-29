from torch import empty, cat, arange
from torch.nn.functional import fold, unfold
import torch
import time

from torch import empty, cat, arange
from torch.nn.functional import fold, unfold
import pickle as pkl

## SGD
class SGD (object):
	def __init__(self, net_param, lr):
		"""
		SGD module declaration
		This class contains a Stocastic Gradient Descend optimizer
		It updates network parameters (attribute net_params) w.r.t their gradients
		THe lerning rate is stored in the attribute lr
		"""
		self.net_param = net_param
		self.lr = lr
	def step(self):
		for param_pair in self.net_param:
			print(param_pair[0])
			param_pair[0].add(param_pair[1].mul(self.lr).mul(-1))
	def zero_grad(self):
		for param_pair in self.net_param:
			param_pair[1].fill_(0.)

## MSE
class MSE (object) :
	def __init__(self):
		"""
		MSE module declaration
		This class contains a Mean Squared Error loss function
		Both forward and backward pass are defined as methods
		The param method does not return any parameter
		"""
	def forward (self, *input) :
		self.difference = input[0]-input[1]
		return (self.difference**2).mean()
	def backward (self):
		factor = self.difference.size(0)*self.difference.size(1)*self.difference.size(2)*self.difference.size(3)
		self.ln = (self.difference.div(factor))*2
		return self.ln
	def param (self) :
		return []

## ReLU
class ReLU (object) :
	def __init__(self):
		"""
		ReLU module declaration
		This class contains a Rectified Linear Unit activation function
		Both forward and backward pass are defined as methods
		The param method does not return any parameter
		The to method allows to map tensor attributes to the processing device
		"""
		self.device = "cpu"
	def forward (self, *input) :
		self.output = empty(input[0].size()).copy_(input[0]).to(device=self.device)
		self.map = self.output<0
		self.output[self.map]=0.
		return self.output
	def backward (self, *gradwrtoutput):
		self.grad_x = empty(self.output.size()).fill_(1.).to(device=self.device)
		self.grad_x[self.map] = 0.
		return self.grad_x.mul(gradwrtoutput[0])
	def param (self) :
		return []
	def to(self, device="cpu"):
		self.device = device

## Sigmoid
class Sigmoid (object) :
	def __init__(self):
		"""
		Sigmoid module declaration
		This class contains a Sigmoid activation function
		Both forward and backward pass are defined as methods
		The param method does not return any parameter
		The to method allows to map tensor attributes to the processing device
		"""
		self.device = "cpu"
	def forward (self, *input) :
		#self.output = empty(input[0].size()).copy_(input[0])
		self.input = empty(input[0].size()).copy_(input[0]).to(device=self.device)
		self.output = self.input.mul(-1).exp().add(1)**(-1)
		return self.output
	def backward (self, *gradwrtoutput):
		self.grad_x = self.input.mul(-1).exp().mul(self.input.mul(-1).exp().add(1)**(-2))
		return self.grad_x.mul(gradwrtoutput[0])
	def param (self) :
		return []
	def to(self, device="cpu"):
		self.device = device

## NNUpsamling
class NNUpsampling (object) :
	def __init__(self, scale_factor):
		"""
		NNUpsampling module declaration
		This module contains a 2D nearest neighbor upsampling algorithm
		The scale attribute stores the ration of output to input sizes
		The param method does not return any parameter
		The to method allows to map tensor attributes to the processing device
		"""
		self.device = "cpu"
		self.scale = scale_factor
	def forward (self, *input) :
		self.input = empty(input[0].size()).copy_(input[0]).to(device=self.device)
		output_size = (
			self.input.size(0),
			self.input.size(1),
			self.input.size(2)*self.scale,
			self.input.size(3)*self.scale,)
		flat =  self.input.view(self.input.size(0)*self.input.size(1),self.input.size(2)*self.input.size(3))
		repeated = flat.unsqueeze(2).repeat(1,1,self.scale**2)
		folded = fold(repeated.transpose(1,2),output_size=(self.input.size(2)*self.scale,self.input.size(3)*self.scale),kernel_size=self.scale,stride=self.scale)
		self.output = folded.view(self.input.size(0),self.input.size(1),folded.size(2),folded.size(3))
		return self.output
	def backward (self, *gradwrtoutput):
		unfolded = unfold(gradwrtoutput[0], kernel_size=self.scale, stride=self.scale)
		self.grad_x = unfolded.view(self.input.size(0),self.input.size(1),self.scale**2,self.input.size(2),self.input.size(3)).sum(dim=2)
		return self.grad_x
	def param (self) :
		return []
	def to(self, device="cpu"):
		self.device = device

## Conv2D
class Conv2d (object) :
	def __init__(self, in_channels, out_channels, kernel_size=(2,2), padding=0, stride=1, dilation=1):
		"""
		Conv2d module
		This class contains a 2D convolution with trainable parameters and weights
		The forward pass performs de 2D convolution on an input using the parameters
		The backward updates the gradient of the parameters and return the gradient of the input
		The param method does not returns pairs of parameters and their gradient
		The to method allows to map tensor attributes to the processing device
		"""
		## arguments as attributes
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.padding = padding
		self.stride = stride
		self.dilation = dilation
		## parameters initialization : from pytorch documentation
		## see https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
		self.k = 1/(self.in_channels*self.kernel_size[0]*self.kernel_size[1])
		self.weight = empty(self.out_channels,self.in_channels, self.kernel_size[0], self.kernel_size[1]).uniform_(-(self.k**0.5), self.k**0.5)
		self.bias = empty(self.out_channels).uniform_(-self.k, self.k)
		## gradient parameters initialization
		self.weight_grad = empty(self.out_channels,self.in_channels, self.kernel_size[0], self.kernel_size[1]).fill_(0.)
		self.bias_grad = empty(self.out_channels).uniform_(-self.k, self.k).fill_(0.)
		## mapping to device
		self.device = "cpu"
		self.weight = self.weight.to(device=self.device)
		self.bias = self.bias.to(device=self.device)
		self.weight_grad = self.weight_grad.to(device=self.device)
		self.bias_grad = self.bias_grad.to(device=self.device)
	def forward (self, *input) :
		## convolution : inspired by the pytorch documentation
		## see https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
		self.input = empty(input[0].size()).copy_(input[0]).to(device=self.device)
		output_H = ((self.input.size(2)+2*self.padding-self.dilation*(self.weight.size(2)-1)-1)/self.stride)+1
		output_W = ((self.input.size(3)+2*self.padding-self.dilation*(self.weight.size(3)-1)-1)/self.stride)+1
		## convolution
		self.unfolded = unfold(self.input, 
			kernel_size=(self.weight.size(2),self.weight.size(3)),
			padding=self.padding,
			stride=self.stride, 
			dilation=self.dilation)
		product =  self.unfolded.transpose(1, 2).matmul(self.weight.view(self.out_channels, -1).t()).transpose(1, 2) + self.bias.view(1, -1, 1)
		self.output = fold(product, (int(output_H), int(output_W)), kernel_size=(1, 1))
		return self.output
	def backward (self, *gradwrtoutput):
		## bias gradient
		self.gradwrtoutput = gradwrtoutput[0]
		self.bias_grad.add(self.gradwrtoutput.sum(dim=(0,2,3)))
		## weight gradient
		temp1 = self.gradwrtoutput.view(self.gradwrtoutput.size(0),self.gradwrtoutput.size(1),-1)
		temp2 = self.unfolded.transpose(1,2)
		product1 = temp1.matmul(temp2)
		self.weight_grad.add(product1.sum(dim=0).view(self.weight_grad.size()))
		## input gradient
		temp3 = self.weight.view(self.out_channels, -1).t()
		product2 = temp3.matmul(temp1)
		x_grad = fold(product2, 
			(self.input.size(2),self.input.size(3)),
			kernel_size=(self.weight.size(2),self.weight.size(3)),
			padding=self.padding,
			stride=self.stride, 
			dilation=self.dilation)
		return x_grad
	def param (self) :
		return [[self.weight, self.weight_grad], [self.bias, self.bias_grad]]
	def to(self, device="cpu"):
		self.device = device
		self.weight = self.weight.to(device=self.device)
		self.bias = self.bias.to(device=self.device)
		self.weight_grad = self.weight_grad.to(device=self.device)
		self.bias_grad = self.bias_grad.to(device=self.device)

## Upsampling
class Upsampling (object):
	def __init__(self, in_channels, out_channels, kernel_size=(2,2), padding=0, stride=1, dilation=1, scale_factor=2):
		"""
		Upsampling module declaration
		This class contains a Conv2D followed by a NNUpsampling layer
		The forward and backward methods perform pass on both layers
		The param method returns the parameters of the Conv2D only as the NNUpsampling has no trainable param
		The to method allows to map tensor attributes to the processing device
		"""
		self.conv2d = Conv2d(in_channels, out_channels, kernel_size, padding, stride, dilation)
		self.nnupsampling = NNUpsampling(scale_factor)
		## mapping to device
		self.device = "cpu"
		self.conv2d.to(device=self.device)
		self.nnupsampling.to(device=self.device)
	def forward (self, *input):
		self.input = input[0]
		self.output = self.conv2d.forward(self.nnupsampling.forward(self.input))
		return self.output
	def backward (self, *gradwrtoutput):
		self.gradwrtoutput = gradwrtoutput[0]
		self.grad = self.nnupsampling.backward(self.conv2d.backward(self.gradwrtoutput))
		return self.grad
	def param(self):
		return self.conv2d.param()
	def to(self, device="cpu"):
		self.device = device
		self.conv2d.to(device=self.device)
		self.nnupsampling.to(device=self.device)

## Sequential
class Sequential (object):
	def __init__(self,*input):
		"""
		Sequential module declaration
		This class contains a sequence made of other modules
		The forward and backward methos perform passes on all modules in the sequence
		The parm method returns the parameters of the modules and their gradient squentially
		The to method allows to map tensor attributes to the processing device
		"""
		self.sequence=[]
		for module in input:
			self.sequence.append(module)
		## mapping to device
		self.device = "cpu"
		for module in self.sequence:
			module.to(device=self.device)
	def append(self,*input):
		for module in input:
			self.sequence.append(module.to(device=self.device))
	def forward (self, *input) :
		x = input[0]
		for module in self.sequence:
			x = module.forward(x)
		self.output = x
		return self.output
	def backward (self, *gradwrtoutput):
		x = gradwrtoutput[0]
		for module in reversed(self.sequence):
			x = module.backward(x)
		self.grad = x
		return self.grad
	def param (self):
		net_params = []
		for module in self.sequence:
			module_params = module.param()
			for params_pair in module_params:
				net_params.append(params_pair)
		return net_params
	def to(self, device="cpu"):
		self.device = device
		for module in self.sequence:
			module = module.to(device=self.device)

## Noise2Noise
class Noise2NoiseModel(object):
	def __init__(self):
		"""
		Noise2Noise model declaration
		The only attribute is the sequence of modules in the model
		Methods include forward and backward passes and parameters netrieval
		The to method allows to map tensor attributes to the processing device
		"""
		self.sequence = Sequential(
			Conv2d(in_channels=3, out_channels=64, kernel_size=(2,2), padding=0, stride=2, dilation=1),
			ReLU(),
			Conv2d(in_channels=64, out_channels=128, kernel_size=(2,2), padding=0, stride=2, dilation=1),
			ReLU(),
			Upsampling(in_channels=128, out_channels=64, kernel_size=(3,3), padding=1, stride=1, dilation=1, scale_factor=2),
			ReLU(),
			Upsampling(in_channels=64, out_channels=3, kernel_size=(3,3), padding=1, stride=1, dilation=1, scale_factor=2),
			Sigmoid(),
		)
		## mapping to device
		self.device = "cpu"
		self.sequence.to(device=self.device)
	def forward (self, *input):
		return self.sequence.forward(input[0])
	def backward (self, *gradwrtoutput):
		return self.sequence.backward(gradwrtoutput[0])
	def param (self):
		return self.sequence.param()
	def to(self, device="cpu"):
		self.device = device
		self.sequence.to(device=self.device)

if __name__ == "__main__":
	x = empty((100,3,28,28)).normal_(0,1)
	y = empty((100,3,28,28)).normal_(0,1)
	net = Noise2NoiseModel()
	optimizer = SGD(net.param(),lr = 0.02)
	criterion = MSE()
	#device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")
	#print(device)
	for device in ["cuda", "cpu"]:
		x = x.to(device=device)
		y = y.to(device=device)
		start = time.time()
		net.to(device=device)
		for epoch in range(100):
			for i in range(10):
				inputs = x[i:i+10]
				targets = y[i:i+10]
				pred = net.forward(inputs)
				loss = criterion.forward(pred,targets)
				net.backward(criterion.backward())
			optimizer.step()
			optimizer.zero_grad()
			print(f"Loss at epoch {epoch+1} : \t {loss}")
		end = time.time()
		print(f"time for device {device} : {end-start}")


 
