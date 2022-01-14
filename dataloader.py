'''eg 100 samples, batch_size=20 --> 100/20=5 iterations for 1 epoch'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import math
from sklearn import datasets
import matplotlib.pyplot as plt
from torchviz import make_dot

class WineDataset(Dataset):

    def __init__(self):
        #data loading
        xy = np.loadtxt('C:/Not C/IIITB/2nd sem/Visual Recognition PE/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

dataset = WineDataset()
# print(dataset[0]) #calls getitem above
# print(len(dataset)) #gets the no of samples in the dataset

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0) #In windows this can cause a problem if set>0, num_worker can make loading faster by using multiple subprocess

#dataiter = iter(dataloader)
# training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
#print(total_samples,n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        if (i+1)%5==0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')