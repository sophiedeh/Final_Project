# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:17:56 2022

@author: 20182672
"""
import torch
from Models_w50to100_hl8to10 import NeuralNetwork

def train(dataloader, model, loss_fn, optimizer):
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
        
        running_loss += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

               
    loss_values_train.append(running_loss/len(dataloader)) #every batch adds to running loss therefore divide by number of batches

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
width = 5
hidlay = 5
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

for i in range(training_times): 
    for u in range(version): 
            # Create two dataloaders
            train_dataloader_SGD = DataLoader(training_data, batch_size=batch_size, shuffle=True,drop_last=True)
            train_dataloader_GD = DataLoader(training_data, batch_size=dataset_size, shuffle=True,drop_last=True)
            
            # Get cpu or gpu device for training.
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using {device} device")
            
            model_SGD = NeuralNetwork(width=width,hidlay=hidlay).to(device)
            model_GD = NeuralNetwork(width=width,hidlay=hidlay).to(device)
            model_SGD = model.load_state_dict(torch.load(f"initial_model_w{width}_hl{hidlay}_v{u+1}.pth", map_location=torch.device(device))) 
            model_GD = model.load_state_dict(torch.load(f"initial_model_w{width}_hl{hidlay}_v{u+1}.pth", map_location=torch.device(device))) 
            
            loss_fn = make_quadratic_hinge_loss()
            optimizer_SGD = torch.optim.SGD(model_SGD.parameters(), lr=1e-3)
            optimizer_GD = torch.optim.GD(model_GD.parameters(), lr=1e-3)
            
            for t in range(epochs):
                print(f"Epoch {t+1}\n-------------------------------")
                train(train_dataloader_SGD, model_SGD, loss_fn, optimizer)
                train(train_dataloader_GD, model_GD, loss_fn, optimizer)
            
            
            # a 1000 steps more, calculate noise strength 
            # 2 version of model, two optimizer (model to optimizer) 
            # 1 do normal training 1 gd
            # one step forward on both models and then difference of their parameters and square and add it up
            losses_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
            losses_GD = 
            
            SGD = losses_train[-1]
            if t   i in range(epochs/100):
                SGDlosses.append(losses_train[(i+1)*(dataset_size/batch_size)])
            
            epoch+=100
        torch.save(MSD,f"MSD_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}.pth")
        epoch = 0
    batch_size += 25