import torch

x=torch.rand(2,2)
y=torch.rand(2,2)

                                ## basic mathematical operations ##
z=x+y
z=torch.add(x,y)
#### Trailing underscore means in-place operation in pytorch, all operatinos can have a trailing underscore. 
#### In-place means that we can modify the varable itself on which operation is applied for example below y is modified
y.add_(x)
z=torch.sub(x,y)
z=torch.mul(x,y)# elementwise multiplication
z=torch.div(x,y)# elementwise division
print(z)
print(y)

                                ## slicing operations ##

x=torch.rand(5,3)
print(x)
print(x[:,1])
print(x[0,:])
print(x[:,1])
print(x[1,1])#for accessing only 1 element
print(x[1,1].item())#item can be used to extract the value stored inside the tensor only if there is 1 value present and not a whole array


                                ## reshaping ##
x=torch.rand(4,4)
print(x)
y=x.view(16)
y=x.view(-1,8)#when we don't know the value of one of the dimension and it is automatically determined
print(y)

                                ## interchange with numpy array ##

import numpy as np

a=torch.ones(5)
b=a.numpy()
print(a,b)
a.add_(1) ## modify both a and b as both point to same meory location
print(a,b)

a=np.ones(5)
b=torch.from_numpy(a)#by default data type is float64
a+=1#modify both a and b as both point to same memory location
print(a,b)

if torch.cuda.is_available():
    device=torch.device("cuda")
    x=torch.ones(5,device=device)
    y=torch.ones(5)
    y=y.to(device)
    z=x+y#will happen on gpu
    #remeber that numpy can't work on gpu so send z into cpu before converting it into numpy
    z=z.to("cpu")
    z=z.numpy()
    z+=1
    print(z)
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_properties(0))

                                ## requires gradient ##
#this type of tensor tells pytorch that it will be optimized later on in the code, for example weights, so these kind of variables must have argument requires_grad as True
x=torch.ones(5,requires_grad=True)
print(x)