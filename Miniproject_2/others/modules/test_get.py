import torch
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

class MyClass(object):
	x = Parameter(torch.tensor([10]))
	def __init__(self):
		self.x = self.x.to(device="cuda")
		y = 5

if __name__ == "__main__":
	M = MyClass()
	M2 = MyClass()
	print(M.x)
	M.x = M.x.mul(3)
	print(M2.x)