import torch
import numpy as np

#below is the implementation of linear regression

x=torch.tensor(1.0)
y=torch.tensor(2.0)

w=torch.tensor(1.0, requires_grad=True)

#forward pass
y_hat = w*x
loss = (y_hat - y)**2
print(loss)

#backward pass
w.retain_grad()
loss.backward()
print(w.grad)
lr=0.1
w=w-(lr*w.grad)

for i in range(10):
    y_hat = w*x
    loss = (y_hat - y)**2
    print(loss)

    w.retain_grad()
    loss.backward()
    w=w-(lr*w.grad)
    print(w)

##below is another implementation with x and y being a vector

x=torch.tensor([1,2,3,4], dtype=torch.float32)
y=torch.tensor([2,4,6,8], dtype=torch.float32)

w=torch.tensor(0.0, dtype=torch.float32,requires_grad=True)

def forward(x,w):
    return w*x

def loss(y,yhat):
    return ((yhat-y)**2).mean()

for i in range(10):
    y_hat = forward(x,w)
    l = loss(y_hat,y)

    w.retain_grad()
    l.backward()
    w=w-(lr*w.grad)
    print("weights: ",w," ,loss: ",l)


x=torch.tensor([1,2,3,4], dtype=torch.float32)
y=torch.tensor([2,4,6,8], dtype=torch.float32)

w=torch.tensor(0.0, dtype=torch.float32,requires_grad=True)

def forward1(x):
    return w*x

for i in range(100):
    y_pred = forward1(x)
    l=loss(y,y_pred)
    l.backward()
    with torch.no_grad():
        w-=0.01*w.grad
    w.grad.zero_()
    print("weights: ",w," ,loss: ",l)