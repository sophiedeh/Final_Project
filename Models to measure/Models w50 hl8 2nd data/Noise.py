# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:17:56 2022

@author: 20182672
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from Models_w50to100_hl8to10 import NeuralNetwork
import matplotlib.pyplot as plt

def train_SGD(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

     # Compute prediction error
        pred = model(X) #torch.Size([200,1]) similar shape to y
        loss = loss_fn(pred, y) #Summing over all elements is in fuction, torch.Size([])
        #break if i want to save all initial losses
        #import pdb; pdb.set_trace()
        
        # Backpropagation
        optimizer.zero_grad() #no accumulation of gradients
        loss.backward() 
        optimizer.step()
        
        gradient_SGD.append(model.grad)
        
        running_loss += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
               
    #loss_values_train.append(running_loss/len(dataloader)) #every batch adds to running loss therefore divide by number of batches

def train_GD(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

     # Compute prediction error
        pred = model(X) #torch.Size([200,1]) similar shape to y
        loss = loss_fn(pred, y) #Summing over all elements is in fuction, torch.Size([])
        #break if i want to save all initial losses
        #import pdb; pdb.set_trace()
        
        # Backpropagation
        optimizer.zero_grad() #no accumulation of gradients
        loss.backward() 
        optimizer.step()
        
        gradient_GD.append(model.grad)
        
        running_loss += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def make_quadratic_hinge_loss():
    
    def quadratic_hinge(output, target):
        ones_tensor = torch.full((len(output),len(output[0])),1)
        delta_mu = torch.sub(ones_tensor,torch.mul(output, target)) #na first time noticed no values below zero above 1 yes so wrong signs
        delta_mu[delta_mu < 0] = 0
        summed = torch.sum(0.5*torch.square(delta_mu))           
        loss = (1/len(output)) * summed
        return loss
    
    return quadratic_hinge 

training_times = 12
width = 50
hidlay = 8
epochs = 5000
version = 4
dataset_size = 1000
epoch = 0
batch_size = 25 
weights = []
initial_weights = []
GDlosses = []
SGDlosses = []

training_data = torch.load('binary_MNIST_pca_train.pt')

for u in range(version): 
    batch_size = 25
    for i in range(training_times):
            noise_per_bs = []
            batch_sizes = []
            # Create two dataloaders
            train_dataloader_SGD = DataLoader(training_data, batch_size=batch_size, shuffle=True,drop_last=True)
            train_dataloader_GD = DataLoader(training_data, batch_size=dataset_size, shuffle=True,drop_last=True)
            
            # Get cpu or gpu device for training.
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using {device} device")
            
            model_SGD = NeuralNetwork(width=width,hidlay=hidlay).to(device)
            model_GD = NeuralNetwork(width=width,hidlay=hidlay).to(device)
            model_SGD = model_SGD.load_state_dict(torch.load(f"initial_model_w{width}_hl{hidlay}_v{u+1}.pth", map_location=torch.device(device))) 
            model_GD = model_GD.load_state_dict(torch.load(f"initial_model_w{width}_hl{hidlay}_v{u+1}.pth", map_location=torch.device(device))) 
            
            loss_fn = make_quadratic_hinge_loss()
            optimizer_SGD = torch.optim.SGD(model_SGD.parameters(), lr=1e-3)
            optimizer_GD = torch.optim.GD(model_GD.parameters(), lr=1e-3)
            
            gradient_SGD = []
            gradient_GD = []
            
            for t in range(epochs):
                print(f"Epoch {t+1}\n-------------------------------")
                train_SGD(train_dataloader_SGD, model_SGD, loss_fn, optimizer_SGD)
                train_GD(train_dataloader_GD, model_GD, loss_fn, optimizer_GD)
            
            sub = torch.sub(gradient_SGD, gradient_GD)
            aver = sub*sub 
            noise_per_bs.append(aver)
            batch_sizes.append(batch_size)
            
            batch_size += 25
    fig = plt.figure()
    ax = fig.subplots()
    fig.suptitle(f"Noise per batch size w{width} hl{hidlay} v{u+1}")
    ax.plot(batch_sizes, noise_per_bs,'-')
    ax.set_xlabel("Batch size (-)")
    ax.set_ylabel("Noise (-)")
    plt.scatter(batch_sizes, noise_per_bs)
    plt.show()
  