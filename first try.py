import torch
x=torch.rand(2,3)#tensor initialized with random values between 0 and 1
x=torch.empty(2,3,2) #empty tensor initialized
x=torch.ones(2,2)
x=torch.zeros(2,2)
x=torch.ones(2,2,dtype=torch.int16)#float or double or int 
x=torch.tensor([2,0.1])
print(x)#another way to create tensor
print(x.dtype)
print(x.size())
#print(torch.cuda.is_available())