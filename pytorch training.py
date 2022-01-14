#1) dsign model (ip, op, forward pass)
#2) construct loss and optimizer
# 3) training Loop
#     - forwars pass: compute predicition
#     - backward pass : gradients
#     - update weights

import torch
from torch.autograd.grad_mode import no_grad
import torch.nn as nn

# x=torch.tensor([1,2,3,4], dtype=torch.float32)
# y=torch.tensor([2,4,6,8], dtype=torch.float32)

# w=torch.tensor(0.0, dtype=torch.float32,requires_grad=True)

# def forward(x):
#     return torch.mul(w,x)

# learn_rate=0.01
# n_iter=100

# loss=nn.MSELoss()
# optimizer = torch.optim.SGD([w], lr=learn_rate)

# for epoch in range(n_iter):
#     #prediction = forward pass
#     y_pred = forward(x)

#     #loss
#     l=loss(y, y_pred)

#     #gradients = backward pass
#     l.backward()

#     #update weights
#     optimizer.step()

#     #emptying gradients
#     optimizer.zero_grad()

#     if (epoch%10==0):
#         print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

# print(f'predition after training: f(5) = {forward(5):.3f}')


X=torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y=torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape #4,1

input_size = n_features
output_size = n_features

# pytorch model
model = nn.Linear(input_size,output_size)

#custom model
class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        #define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def  forward(self,x):
        return self.lin(x)

model = LinearRegression(input_size,output_size)

print(f'predition before training: f(5) = {model(X_test).item():.3f}')

learn_rate=0.01
n_iter=100

loss=nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

for epoch in range(n_iter):
    #prediction = forward pass
    y_pred = model(X)

    #loss
    l=loss(Y, y_pred)

    #gradients = backward pass
    l.backward()

    #update weights
    optimizer.step()

    #emptying gradients
    optimizer.zero_grad()

    if (epoch%10==0):
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'predition after training: f(5) = {model(X_test).item():.3f}')