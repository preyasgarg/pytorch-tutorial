import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchviz import make_dot


#prepare dataset
bc = datasets.load_breast_cancer()
x, y = bc.data,bc.target

n_samples, n_features = x.shape

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

#scale
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train=torch.from_numpy(X_train.astype(np.float32))
y_train=torch.from_numpy(y_train.astype(np.float32))
X_test=torch.from_numpy(X_test.astype(np.float32))
y_test=torch.from_numpy(y_test.astype(np.float32))

y_train=y_train.view(y_train.shape[0],1)
y_test=y_test.view(y_test.shape[0],1)


#custom model
class LogisticRegression(nn.Module):

    def __init__(self, n_inp_features):
        super(LogisticRegression, self).__init__()
        #define layers
        self.lin = nn.Linear(n_inp_features, 1)

    def  forward(self,x):
        y_predicted = torch.sigmoid(self.lin(x))
        return y_predicted

model = LogisticRegression(n_features)


#loss and optimizer
learn_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

#training loop
iterations = 101
for epoch in range(iterations):
    #forward pass and loss
    y_predicted = model(X_train)
    loss=criterion(y_predicted,y_train)

    #backward pass
    loss.backward()

    #update
    optimizer.step()

    #emptying the gradients here
    optimizer.zero_grad()

    if (epoch+1)%10==0:
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {loss:.8f}')# this weight here is to be looked into


make_dot(loss).render("logistic regression", format="png") #will be saved inside c:/users/preyas, visualize computation graph

with torch.no_grad():
    y_predicted=model(X_test)
    y_predicted_cls = y_predicted.round()#as class labels are binary so we do this as greater than 5 is assigned 1 and less as 0
    acc = y_predicted_cls.eq(y_test).sum()/float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')

#plot
plt.plot(X_test,y_predicted_cls,'ro')
plt.plot(X_test,y_test,'go')
plt.show()