import torch
from torch import empty, arange, cat

"""class Parameter (object):
	def __init__(self, value):
		self.value = value
	def set_value(self,new_value):
		self.value = new_value
	def add_value(self, new_value):
		self.value = self.value+new_value
	def zero_value(self):
		self.value = self.value.fill_(0.)"""

class Parameter(object):
    def __init__(self, value):
        self.device = "cpu"
        self.value = value.to(device=self.device)

    def __get__(self, obj, objtype):
        return self.value

    def __set__(self, obj, value):
        self.value = value.to(device=self.device)

    def to(self, device):
    	self.device = device
    	self.value = self.value.to(device=device)

x = empty((1,3,4,4)).normal_(0,1)
param = Parameter(x)
print(param.value)