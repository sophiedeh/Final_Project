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
        # new function called calculation noise has at the end of for loop break
        optimizer.zero_grad() #no accumulation of gradients
        loss.backward()
        optimizer.step()
        #time_step_SGD += 1
        #import pdb; pdb.set_trace()
        
        for name, param in model.named_parameters():
            for i in range(hidlay*2+1):
                if name == f"linear_relu_stack.{i}.weight":
                    gradient_SGD = param.grad 


        running_loss += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        break #since we break after the first step the number of time steps is equal to epoch
    return gradient_SGD#, time_step_SGD
               
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
        
        for name, param in model.named_parameters():
            for i in range(hidlay*2+1):
                if name == f"linear_relu_stack.{i}.weight":
                    gradient_GD = param.grad # should return 
                    
        running_loss += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    return gradient_GD  

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
width = 10
hidlay = 6
epochs = 10000 #escape time is from above 20000 number of steps so should be in initial plateau - 10000
version = 4
dataset_size = 1000
epoch = 0
batch_size = 25 
weights = []
initial_weights = []
GDlosses = []
SGDlosses = []
#time_step_SGD = 0
#time_step_GD = 0

training_data = torch.load('binary_MNIST_pca_train.pt')

for u in range(version): 
    batch_size = 25
    noise_per_bs = []
    batch_sizes = []
    for i in range(training_times):
            # Create two dataloaders
            train_dataloader_SGD = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
            train_dataloader_GD = DataLoader(training_data, batch_size=dataset_size, shuffle=True, drop_last=True)
            
            # Get cpu or gpu device for training.
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using {device} device")
            
            modelSGD = NeuralNetwork(width=width,hidlay=hidlay).to(device)
            modelGD = NeuralNetwork(width=width,hidlay=hidlay).to(device)
            modelSGD.load_state_dict(torch.load(f"initial_model_w{width}_hl{hidlay}_v{u+1}.pth", map_location=torch.device(device))) 
            modelGD.load_state_dict(torch.load(f"initial_model_w{width}_hl{hidlay}_v{u+1}.pth", map_location=torch.device(device)))
            
            loss_fn = make_quadratic_hinge_loss()
            optimizerSGD = torch.optim.SGD(modelSGD.parameters(), lr=1e-3)
            optimizerGD = torch.optim.SGD(modelGD.parameters(), lr=1e-3)
            
            gradient_SGD = []
            gradient_GD = []
            time_step_GD = 0
            time_step_SGD = 0
            
            for t in range(epochs):
                #if time_step_GD != 120000 and time_step_SGD != 120000:
                    print(f"Epoch {t+1}\n-------------------------------")
                    #train_SGD(train_dataloader_SGD, modelSGD, loss_fn, optimizerSGD)
                    #train_GD(train_dataloader_GD, modelGD, loss_fn, optimizerGD)
                    # a,b = train_SGD(train_dataloader_SGD, modelSGD, loss_fn, optimizerSGD)
                    # c,d = train_GD(train_dataloader_GD, modelGD, loss_fn, optimizerGD)
                    gradient_SGD.append(train_SGD(train_dataloader_SGD, modelSGD, loss_fn, optimizerSGD))
                    gradient_GD.append(train_GD(train_dataloader_GD, modelGD, loss_fn, optimizerGD))
                    # time_step_GD += d
                    # time_step_SGD += b
                
            aver_noises = []
            for i in range(len(gradient_SGD)):
                sub = torch.sub(gradient_SGD[i],gradient_GD[i]) #per weight of that layer determine difference - I did SGD - GD does not make difference with squaring?
                noise = torch.square(sub) #noise is difference*difference
                aver_noise = torch.mean(noise) #average noise over the weights of that layer
                aver_noises.append(aver_noise)
            
            aver_noises = torch.DoubleTensor(aver_noises)
            aver_noise_all_layers = torch.mean(aver_noises)

            # Save average noise over all the layers for this batch size
            noise_per_bs.append(aver_noise_all_layers)
            batch_sizes.append(batch_size)
            
            batch_size += 25
    fig = plt.figure()
    ax = fig.subplots()
    fig.suptitle(f"Noise per batch size w{width} hl{hidlay} v{u+1}")
    ax.plot(batch_sizes, noise_per_bs,'-', marker='o')
    ax.set_xlabel("Batch size (-)")
    ax.set_ylabel("Noise (-)")
    #plt.scatter(batch_sizes, noise_per_bs)
    plt.show()
    plt.savefig(f"Noise per batch size w{width} hl{hidlay} v{u+1}.png",bbox_inches='tight')
  