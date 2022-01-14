import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from torchviz import make_dot


#prepare data
x_np, y_np = datasets.make_regression(n_samples=100, n_features=1,noise=20,random_state=1)

X=torch.from_numpy(x_np.astype(np.float32))
y=torch.from_numpy(y_np.astype(np.float32))
y=y.view(y.shape[0],1)

n_samples, n_features = X.shape

#model
ip_size = n_features
op_size = 1
model = nn.Linear(ip_size,op_size)

#loss and optimizer
learn_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

#training loop
iterations = 100
for epoch in range(iterations):
    #forward pass and loss
    y_predicted = model(X)
    loss=criterion(y_predicted,y)

    #backward pass
    loss.backward()

    #update
    optimizer.step()

    optimizer.zero_grad()

    if (epoch%10==0):
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {loss:.8f}')# this weight here is to be looked into


make_dot(loss).render("detached", format="png") #will be saved inside c:/users/preyas, visualize computation graph

#plot
predicted = model(X).detach().numpy() #detach removes this operation from being part of computation graph, subgraph involving this view or tensor is not recorded
plt.plot(X,y,'ro')
plt.plot(X,predicted,'b')
plt.show()