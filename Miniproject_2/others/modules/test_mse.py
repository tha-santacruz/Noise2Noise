import torch
from torch import empty, arange, cat


class MSE (object) :
	def __init__(self):
		"""MSE module declaration"""
	def forward (self, *input) :
		self.difference = input[0]-input[1]
		return (self.difference**2).mean()
	def backward (self):
		factor = self.difference.size(0)*self.difference.size(1)*self.difference.size(2)*self.difference.size(3)
		self.ln = (self.difference.div(factor))*2
		return self.ln
	def param (self) :
		return []

#x = empty((1,3,4,4)).normal_(0,1)
x = torch.randint(-2,2,(10,3,4,4)).to(torch.float32)
x.requires_grad = True
print(x)

#y = empty((1,3,4,4)).normal_(0,1)
y = torch.randint(-2,2,(10,3,4,4)).to(torch.float32)
print(y)

torch_MSE = torch.nn.MSELoss()
my_MSE = MSE()


torch_out = torch_MSE(x,y)
torch_out.backward()
torch_grad = x.grad #.mul(torch_out)
my_out = my_MSE.forward(x,y)
print("outputs")
print(f"torch {torch_out}")
print(f"mine {my_out}")

torch.testing.assert_allclose(my_out , torch_out)
print("same output")

my_grad = my_MSE.backward()

print(torch_grad)
print(torch_grad.sum())
print(my_grad)
print(my_grad.sum())
torch.testing.assert_allclose(my_grad , torch_grad)
print("same gradient")