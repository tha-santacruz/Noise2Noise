from torch import empty, cat, arange
from torch.nn.functional import fold, unfold
import pickle as pkl
from pathlib import Path


## Parameter
class Parameter (object):
	"""
	Parameter class
	This class contains a parameter that stores a tensor as a value
	Three methods allow to update the parameter value
	The to method allows to map tensor attributes to the processing device
	"""
	def __init__(self, value):
		self.device = "cpu"
		self.value = value.to(device=self.device)
	def set_value(self,new_value):
		self.value = new_value
	def add_value(self, new_value):
		self.value = self.value+new_value
	def zero_value(self):
		self.value = self.value.fill_(0.)
	def to(self, device="cpu"):
		self.device = device
		self.value = self.value.to(device=self.device)

## SGD
class SGD (object):
	"""
	SGD module
	This class contains a Stocastic Gradient Descend optimizer
	It updates network parameters (attribute net_params) w.r.t their gradients
	The lerning rate is stored in the attribute lr
	"""
	def __init__(self, net_param, lr):
		self.net_param = net_param
		self.lr = lr
	def step(self):
		for param_pair in self.net_param:
			param_pair[0].add_value(param_pair[1].value.mul(self.lr).mul(-1))
	def zero_grad(self):
		for param_pair in self.net_param:
			param_pair[1].zero_value()

## MSE
class MSE (object) :
	"""
	MSE module
	This class contains a Mean Squared Error loss function
	Both forward and backward pass are defined as methods
	The param method does not return any parameter
	"""
	def __init__(self):
		pass
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
	"""
	ReLU module
	This class contains a Rectified Linear Unit activation function
	Both forward and backward pass are defined as methods
	The param method does not return any parameter
	The to method allows to map tensor attributes to the processing device
	"""
	def __init__(self):
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
	"""
	Sigmoid module
	This class contains a Sigmoid activation function
	Both forward and backward pass are defined as methods
	The param method does not return any parameter
	The to method allows to map tensor attributes to the processing device
	"""
	def __init__(self):
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
	"""
	NNUpsampling module
	This module contains a 2D nearest neighbor upsampling algorithm
	The scale attribute stores the ration of output to input sizes
	The param method does not return any parameter
	The to method allows to map tensor attributes to the processing device
	"""
	def __init__(self, scale_factor):
		self.device = "cpu"
		self.scale = scale_factor
	def forward (self, *input) :
		self.input = empty(input[0].size()).copy_(input[0]).to(device=self.device)
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
	"""
	Conv2d module
	This class contains a 2D convolution with trainable parameters and weights
	The forward pass performs de 2D convolution on an input using the parameters
	The backward updates the gradient of the parameters and return the gradient of the input
	The param method does not returns pairs of parameters and their gradient
	The to method allows to map tensor attributes to the processing device
	"""
	def __init__(self, in_channels, out_channels, kernel_size=2, padding=0, stride=1, dilation=1):
		## arguments as attributes
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.padding = padding
		self.stride = stride
		self.dilation = dilation
		## parameters initialization : from pytorch documentation
		## see https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
		self.k = 1/(self.in_channels*self.kernel_size*self.kernel_size)
		self.weight_p = Parameter(empty(self.out_channels,self.in_channels, self.kernel_size, self.kernel_size).uniform_(-(self.k**0.5), self.k**0.5))
		self.bias_p = Parameter(empty(self.out_channels).uniform_(-self.k, self.k))
		## gradient parameters initialization
		self.weight_p_grad = Parameter(empty(self.out_channels,self.in_channels, self.kernel_size, self.kernel_size).fill_(0.))
		self.bias_p_grad = Parameter(empty(self.out_channels).uniform_(-self.k, self.k).fill_(0.))
		## mapping to device
		self.device = "cpu"
		self.weight_p.to(device=self.device)
		self.bias_p.to(device=self.device)
		self.weight_p_grad.to(device=self.device)
		self.bias_p_grad.to(device=self.device)
		## parameters values (for test.py file)
		self.weight = self.weight_p.value
		self.bias = self.bias_p.value
	def forward (self, *input) :
		## convolution : inspired by the pytorch documentation
		## see https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
		self.input = empty(input[0].size()).copy_(input[0]).to(device=self.device)
		output_H = ((self.input.size(2)+2*self.padding-self.dilation*(self.weight_p.value.size(2)-1)-1)/self.stride)+1
		output_W = ((self.input.size(3)+2*self.padding-self.dilation*(self.weight_p.value.size(3)-1)-1)/self.stride)+1
		## convolution
		self.unfolded = unfold(self.input, 
			kernel_size=(self.weight_p.value.size(2),self.weight_p.value.size(3)),
			padding=self.padding,
			stride=self.stride, 
			dilation=self.dilation)
		product =  self.unfolded.transpose(1, 2).matmul(self.weight_p.value.view(self.out_channels, -1).t()).transpose(1, 2) + self.bias_p.value.view(1, -1, 1)
		self.output = fold(product, (int(output_H), int(output_W)), kernel_size=(1, 1))
		return self.output
	def backward (self, *gradwrtoutput):
		## bias gradient
		self.gradwrtoutput = gradwrtoutput[0]
		self.bias_p_grad.add_value(self.gradwrtoutput.sum(dim=(0,2,3)))
		## weight gradient
		temp1 = self.gradwrtoutput.view(self.gradwrtoutput.size(0),self.gradwrtoutput.size(1),-1)
		temp2 = self.unfolded.transpose(1,2)
		product1 = temp1.matmul(temp2)
		self.weight_p_grad.add_value(product1.sum(dim=0).view(self.weight_p_grad.value.size()))
		## input gradient
		temp3 = self.weight_p.value.view(self.out_channels, -1).t()
		product2 = temp3.matmul(temp1)
		x_grad = fold(product2, 
			(self.input.size(2),self.input.size(3)),
			kernel_size=(self.weight_p.value.size(2),self.weight_p.value.size(3)),
			padding=self.padding,
			stride=self.stride, 
			dilation=self.dilation)
		return x_grad
	def param (self) :
		return [[self.weight_p, self.weight_p_grad], [self.bias_p, self.bias_p_grad]]
	def to(self, device="cpu"):
		self.device = device
		self.weight_p.to(device=self.device)
		self.bias_p.to(device=self.device)
		self.weight_p_grad.to(device=self.device)
		self.bias_p_grad.to(device=self.device)

## Upsampling
class Upsampling (object):
	"""
	Upsampling module
	This class contains a Conv2D followed by a NNUpsampling layer
	The forward and backward methods perform pass on both layers
	The param method returns the parameters of the Conv2D only as the NNUpsampling has no trainable param
	The to method allows to map tensor attributes to the processing device
	"""
	def __init__(self, in_channels, out_channels, kernel_size=2, padding=0, stride=1, dilation=1, scale_factor=2):
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
	"""
	Sequential module
	This class contains a sequence made of other modules
	The forward and backward methos perform passes on all modules in the sequence
	The parm method returns the parameters of the modules and their gradient squentially
	The to method allows to map tensor attributes to the processing device
	"""
	def __init__(self,*input):
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
	"""
	Noise2Noise model
	The only attribute is the sequence of modules in the model
	Methods include forward and backward passes and parameters netrieval
	The to method allows to map tensor attributes to the processing device
	"""
	def __init__(self):
		self.sequence = Sequential(
			Conv2d(in_channels=3, out_channels=64, kernel_size=6, padding=2, stride=2, dilation=1),
			ReLU(),
			Conv2d(in_channels=64, out_channels=128, kernel_size=6, padding=2, stride=2, dilation=1),
			ReLU(),
			Upsampling(in_channels=128, out_channels=64, kernel_size=7, padding=3, stride=1, dilation=1, scale_factor=2),
			ReLU(),
			Upsampling(in_channels=64, out_channels=3, kernel_size=7, padding=3, stride=1, dilation=1, scale_factor=2),
			Sigmoid(),
		)
		## mapping to device
		self.device = "cpu"
		self.sequence.to(device=self.device)
	def forward (self, *input):
		return self.sequence.forward(input[0]).mul(255.)
	def backward (self, *gradwrtoutput):
		return self.sequence.backward(gradwrtoutput[0].div(255))
	def param (self):
		return self.sequence.param()
	def to(self, device="cpu"):
		self.device = device
		self.sequence.to(device=self.device)

class Model():
	"""
	Model declaration with related methods. Includes :
	- Pretrained model parameters loading
	- Trained model saving
	- Model training
	- Prediction with the model's current state
	- Data augmentation for training
	"""
	def __init__(self) -> None:
		## Training batch size
		self.BATCH_SIZE = 10
		## Data augmentation for training
		self.augment = True

		## Model, loss function and optimizer declaration
		self.net = Noise2NoiseModel()
		self.optimizer = SGD(self.net.param(), lr=0.01)
		self.criterion = MSE()

		## Processing device
		try:
			self.device = "cuda"
			self.net.to(device=self.device)
		except:
			self.device = "cpu"
			self.net.to(device=self.device)

	def load_pretrained_model(self, model_name = "bestmodel.pkl") -> None:
		## This loads the parameters saved in bestmodel.pth into the model
		model_path = Path(__file__).parent / model_name
		with open(model_path, "rb") as file:
			param_values = pkl.load(file)
		net_params = self.net.param()
		for i in range(len(net_params)):
			net_params[i][0].set_value(param_values[i][0])
			net_params[i][0].to(device=self.device)
			net_params[i][1].set_value(param_values[i][1])
			net_params[i][1].to(device=self.device)

	def save_trained_model(self, model_name = "newmodel.pkl") -> None:
		## This saves the parameters of the model
		param_values = []
		for param_pair in self.net.param():
			param_values.append([param_pair[0].value, param_pair[1].value])
		model_path = Path(__file__).parent / model_name
		with open(model_path, "wb") as file:
			pkl.dump(param_values, file)

	def train(self, train_input, train_target, num_epochs):
		## train_input: tensor of size (N, C, H, W) containing a noisy version of the images
		## train_target: tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs from the input by their noise
		train_input = train_input.to(device=self.device).float()
		train_target = train_target.to(device=self.device).float()
		## Predicting and back propagating for each batch at each epoch
		for epoch in range(num_epochs):
			for b in range(0, train_input.size(0), self.BATCH_SIZE):
				input = train_input.narrow(0, b, self.BATCH_SIZE)
				target = train_target.narrow(0, b, self.BATCH_SIZE)
				if self.augment:
					input, target = self.augment_data(train_input=input, train_target=target)
				pred = self.net.forward(input)
				loss = self.criterion.forward(pred,target)
				self.net.backward(self.criterion.backward())
				self.optimizer.step()
				self.optimizer.zero_grad()

	def predict(self, test_input): #-> torch.Tensor:
		## test_input : tensor of size (N1, C, H, W) that has to be denoised by the trained or loaded network
		## Returns a tensor of the size (N1, C, H, W)
		test_input = test_input.to(device=self.device)
		return self.net.forward(test_input)

	def augment_data(self, train_input, train_target, noise_part=0.1):
		# This method allows to augment training data by applying simple random transformations at the batch level
		# Gaussian noise with mean = 0 and with std = noise_part*data_std
		noise_std = cat((train_input,train_target),0).std()*noise_part
		input_noise = empty(train_input.size()).normal_(mean=0,std=noise_std).to(device=self.device).float()
		train_input = train_input+input_noise
		train_input[train_input<0] = 0.0
		train_input[train_input>255] = 255.0
		## The four lines below can be uncommented to apply random noise on targets too.
		#target_noise = empty(train_target.size()).normal_(mean=0,std=noise_std).to(device=self.device).float()
		#train_target = train_target+target_noise
		#train_target[train_target<0] = 0.0
		#train_target[train_target>255] = 255.0
		## Random transpose
		transpose = empty((1)).random_(0,2).bool()
		if transpose:
			train_input = train_input.transpose(-1, -2)
			train_target = train_target.transpose(-1, -2)
		## Random rotation
		num_rot = empty((1)).random_(0,4).int()
		train_input= train_input.rot90(num_rot.item(), [-1, -2])
		train_target= train_target.rot90(num_rot.item(), [-1, -2])
		return train_input, train_target