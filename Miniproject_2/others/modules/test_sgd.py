from torch import empty, arange, cat
from torch.nn.functional import unfold, fold

class SGD (object):
	def __init__(self, net_param, lr):
		"""SGD module declaration
		This class contains a Stockastic Gradient Descend optimizer
		It updates network parameters (attribute net_params) w.r.t their gradients
		THe lerning rate is stored in the attribute lr"""
		self.net_param = net_param
		self.lr = lr
	def step(self):
		for param_pair in self.net_param:
			param_pair[0].add_value(param_pair[1].value.mul(self.lr))
	def zero_grad(self):
		for param_pair in self.net_param:
			param_pair[1].zero_value()


class MSE (object) :
	def __init__(self):
		"""MSE module declaration
		This class contains a Mean Squared Error loss function
		Both forward and backward pass are defined as methods
		The param methods does not return any parameter"""
	def forward (self, *input) :
		self.difference = input[0]-input[1]
		return (self.difference**2).mean()
	def backward (self):
		factor = self.difference.size(0)*self.difference.size(1)*self.difference.size(2)*self.difference.size(3)
		self.ln = (self.difference.div(factor))*2
		return self.ln
	def param (self) :
		return []

class Parameter (object):
	def __init__(self, value):
		self.value = value
	def set_value(self,new_value):
		self.value = new_value
	def add_value(self, new_value):
		self.value = self.value+new_value
		print("parameter updated")
	def zero_value(self):
		self.value = self.value.fill_(0.)
		print("parameter zeroed")

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
		self.grad_bias.add_value(gradwrtoutput[0].sum())
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

x = empty((10,3,4,4)).normal_(0,1)
y = empty((10,3,4,4)).normal_(0,1)
#print(x)


dummy_model = Sequential(DummyModule(3,4,4),DummyModule(3,4,4),Sigmoid())
optimizer = SGD(dummy_model.param(),0.02)
print("##########################")
print("prediction")
pred = dummy_model.forward(x)
criterion = MSE()
loss = criterion.forward(pred,y)
print(loss)
print("##########################")
print("gradients before backward")
for params_pair in dummy_model.param():
	print(params_pair[1].value)
gradient = dummy_model.backward(criterion.backward())
print("##########################")
print("gradients after backward")
for params_pair in dummy_model.param():
	print(params_pair[1].value)

print("##########################")
print("params before update")
for params_pair in dummy_model.param():
	print(params_pair[0].value)
optimizer.step()
print("##########################")
print("params after update")
for params_pair in dummy_model.param():
	print(params_pair[0].value)

optimizer.zero_grad()
print("##########################")
print("gradients after zero_grad")
for params_pair in dummy_model.param():
	print(params_pair[1].value)
















