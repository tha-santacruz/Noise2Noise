import torch
from torch import empty, arange, cat

class Parameter (object):
	def __init__(self, value):
		self.value = value
	def set_value(self,new_value):
		self.value = new_value
	def add_value(self, new_value):
		self.value = self.value+new_value
	def zero_value(self):
		self.value = self.value.fill_(0.)

class Sigmoid (object) :
	def __init__(self):
		"""ReLU module declaration"""
	def forward (self, *input):
		#self.output = empty(input[0].size()).copy_(input[0])
		self.input = empty(input[0].size()).copy_(input[0])
		self.output = self.input.mul(-1).exp().add(1)**(-1)
		return self.output
	def backward (self, *gradwrtoutput):
		self.grad_x = self.input.mul(-1).exp().mul(self.input.mul(-1).exp().add(1)**(-2))
		return self.grad_x.mul(gradwrtoutput[0])
	def param (self):
		return []

## dummy module
class DummyModule (object):
	def __init__(self, *input_size):
		print("initialized dummy module")
		self.params = Parameter(empty((input_size)).uniform_(-0.5, 0.5))
		self.grad_params = Parameter(empty((input_size)).fill_(0.))
		self.bias = Parameter(empty((1)).fill_(0.))
		self.grad_bias = Parameter(empty((1)).fill_(0.))
	def forward (self, *input) :
		self.input = input[0]
		self.output = self.input.mul(self.params.value)
		return self.output
	def backward (self, *gradwrtoutput):
		self.grad_input = gradwrtoutput[0].mul(self.params.value) + self.bias.value
		self.grad_params.add_value(self.input.mul(gradwrtoutput[0]).sum(dim=0))
		self.grad_bias.add_value(gradwrtoutput[0].sum((1,2,3)))
		return self.grad_input
	def param (self):
		return [[self.params, self.grad_params],[self.bias, self.grad_bias]]

class Sequential (object):
	def __init__(self,*input):
		"""Sequential module declaration"""
		self.sequence=[]
		for module in input:
			self.sequence.append(module)
	def append(self,*input):
		for module in input:
			self.sequence.append(module)
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

x = empty((1,3,4,4)).normal_(0,1)
x.requires_grad = True
#print(x)


dummy_model = Sequential(DummyModule(3,4,4),DummyModule(3,4,4),Sigmoid())
params = dummy_model.param()
print("##########################")
print(f"sequence : {dummy_model.sequence}")
print(f"param : {dummy_model.param()[1]}")
print(len(dummy_model.param()))

pair = dummy_model.param()[1]
print("##########################")
print(pair[0].value)
pair[0].add_value(12) 
print(pair[0].value)
print("##########################")

print(f"param : {dummy_model.param()[1][0].value}")

print("##########################")
output = dummy_model.forward(x)
print("forward")
print(output)
print("##########################")
print("backward")
back = dummy_model.backward(output)
print(back)

"""torch_relu = torch.nn.ReLU()
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
print("same gradient")"""