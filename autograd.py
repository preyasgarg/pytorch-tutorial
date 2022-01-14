import torch
x=torch.randn(3, requires_grad=True)
print(x)

y=x+2 #grad fumction is add backwards
z=y*y*2 #grad fumction is multiply backwards
y=y.mean()
z=z.mean() #grad fumction is mean backwards
print(y)
print(z)

z.backward()
#y.backward()
print(x.grad)

## most of time output is scalar value, in case it is not then z.backward() won't work, to make it work we have to pass it a tensor 
y=x+2 #grad fumction is add backwards
z=y*y*2
v=torch.tensor([0.1,1.0,0.001],dtype=torch.float32)
z.backward(v) #dz/dx
print(x.grad)
print(v)

##  sometimes we don't want want pytorch to track operations in computational graph of variable or tensor to not a part of backprop or gradient functions, for example updating weights. ##
##  so for that we use some functions ##
x=torch.randn(3,requires_grad=True)
print(x)
# x.requires_grad_(False)#as trailing underscore so modify tensor/variable in-place
# print(x)
# y=x.detach()
# print(y)
with torch.no_grad():
    y=x+2
    print(y)

## empty the gradient before going for next iteration in loop #3
weights=torch.ones(4,requires_grad=True)

for i in range(3):
    model_output = (weights*3).sum()

    model_output.backward()

    ## below will print wrong gradients as accumulated across iterations so we must empty it
    print(weights.grad) ## remeber that gradients are stored inside variables.grad

    ## upon using below, our gradients will be emptied out after each iteration
    weights.grad.zero_()##emptying the gradients

ws = torch.ones(4, requires_grad=True)
optimizer = torch.optim.SGD([ws], lr=0.01)
optimizer.step() 
optimizer.zero_grad() #reseting the gradients or emptying the gradients


## thing to summarize:
# whenever we want calculate the gradient we must specify requires_grad=True
# to calculate gradient use backward function
# before next step in our optimization we must empty our gradients 
# we must also know how to not allow some operation to be a part of computation graph