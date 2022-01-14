import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x= np.array([2.0,1.0,0.1])
outputs = softmax(x)
print('softmax numpy:', outputs)

x=torch.tensor([2.0,1.0,0.1])
outputs=torch.softmax(x, dim=0)
print(outputs)


def cross_entropy(actual,predicted):
    loss= -np.sum(actual * np.log(predicted))
    return loss

y= np.array([1,0,0])

y_pred_good = np.array([0.7,0.2,0.1])
y_pred_bad = np.array([0.1,0.3,0.6])
l1 = cross_entropy(y,y_pred_good)
l2 = cross_entropy(y,y_pred_bad)
print(f'loss1 numpy: {l1:.4f}')
print(f'loss2 numpy: {l2:.4f}')


#nn.CLE is nn.logsoftmax + nn.nllloss so don't apply softmax before cc.CLE i.e. no softmax in last layer
#y has class labels not one hot encoded, y_pred has raw scores(logits), no softmax
loss = nn.CrossEntropyLoss()

y=torch.tensor([0]) #correct class label is 0th class

#nsamples x nclasses = 1x3
y_pred_good = torch.tensor([[2.0,1.0,0.1]]) #good prediction as 0th class has highest score
y_pred_bad = torch.tensor([[0.5,1.6,0.4]]) #good prediction as 0th class has highest score
l1=loss(y_pred_good,y)
l2=loss(y_pred_bad,y)
#this prints the loss
print(l1.item(),l2.item())

#to get class predictions
_, predictions1 = torch.max(y_pred_good,1) #1 represents the dimension to be searched for
_, predictions2 = torch.max(y_pred_bad,1)
print(predictions1,predictions2)





#let's look at multiple l=samples loss

y=torch.tensor([2,0,1]) #correct class label is 0th class

#nsamples x nclasses = 1x3
y_pred_good = torch.tensor([[0.1,1.0,2.1],[2.0,1.0,0.1],[0.1,3.0,0.1]]) #good prediction as 0th class has highest score
y_pred_bad = torch.tensor([[2.1,1.0,0.1],[0.1,1.0,2.1],[0.1,3.0,0.1]]) #good prediction as 0th class has highest score
l1=loss(y_pred_good,y)
l2=loss(y_pred_bad,y)
#this prints the loss
print(l1.item(),l2.item())

#to get class predictions
_, predictions1 = torch.max(y_pred_good,1) #1 represents the dimension to be searched for
_, predictions2 = torch.max(y_pred_bad,1)
print(predictions1,predictions2)