# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 09:20:47 2022

@author: 20182672
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import numpy as np
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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

               
    loss_values_train.append(running_loss/len(dataloader)) #every batch adds to running loss therefore divide by number of batches
                     
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            tensor_multiplied = torch.mul(pred,y)
            bigger_than_one = tensor_multiplied[tensor_multiplied>1]
            correct += torch.numel(bigger_than_one) #equal sign means positive and other requirement is bigger than 1   
    test_loss /= num_batches
    loss_values_test.append(test_loss)
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def make_quadratic_hinge_loss():
    
    def quadratic_hinge(output, target):
        ones_tensor = torch.full((len(output),len(output[0])),1)
        delta_mu = torch.sub(ones_tensor,torch.mul(output, target)) #na first time noticed no values below zero above 1 yes so wrong signs
        delta_mu[delta_mu < 0] = 0
        summed = torch.sum(0.5*torch.square(delta_mu))           
        loss = (1/len(output)) * summed
        return loss
    
    return quadratic_hinge    

training_times = 12 #amount of how many times to train data
width = 50 #amount of nodes per layer
hidlay = 8 #amount of layers
different_depth = 1
different_width = 1

# Load training data from own script. 
training_data = torch.load('binary_MNIST_pca_train_2nd_version.pt') #Tensordataset with first image and then 1 or -1

# Load test data from own script.
test_data = torch.load('binary_MNIST_pca_test_2nd_version.pt')

# Define size 
dataset_size = len(training_data)

# Define the batch size
batch_size = 25

version = 4
for i in range(training_times): 
    for u in range(version):
        # Create data loaders.
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True,drop_last=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

        for X, y in test_dataloader:
            print(f"Shape of X [N, C, H]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break
    
        # Get cpu or gpu device for training.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")
        
        model = NeuralNetwork(width=width,hidlay=hidlay).to(device)
        model.load_state_dict(torch.load(f"initial_model_w{width}_hl{hidlay}_v{u+1}.pth", map_location=torch.device(device))) 
        #model.load_state_dict(torch.load(f"initial_model_w{width}_hl{hidlay}.pth")) 
        
        loss_fn = make_quadratic_hinge_loss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        
        loss_values_train = []
        loss_values_test = []
      
        start = time.time()
        epochs = 50000
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            test(test_dataloader, model, loss_fn)
            end = time.time()
            elapsed_time = end - start
            
            if t % 100 == 0:
                torch.save(model.state_dict(), f"model_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{t}_tt{training_times}.pth")
                print(f"Saved PyTorch Model State to model_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{t}_tt{training_times}.pth")
        
        print("Done!")
        print(f"Elapsed time {elapsed_time}\n-------------------------------")
            
        number_of_steps = []
        for i in range(len(loss_values_train)):
            number_of_steps.append((i+1)*(dataset_size/batch_size)) # i + 1 since first value corresponds to first epoch so not zero epoch
            
        fig, axs = plt.subplots(3,sharex=True)
        fig.suptitle(f"Train vs Test Losses width{width} hidlay{hidlay} version{u+1}")
        axs[0].plot(number_of_steps, loss_values_train,'-')
        axs[0].set_title('Train log')
        axs[1].plot(number_of_steps, loss_values_train,'-')
        axs[1].set_title('Train linear')
        axs[2].plot(number_of_steps, loss_values_test,'-')
        axs[2].set_title('Test')
    
        for ax in axs.flat:
            ax.set(xlabel='Number of steps', ylabel='Loss')
            ax.set_yscale('log')
    
        for ax in axs.flat:
            ax.label_outer()
            
        axs[1].set_yscale('linear')
    
        plt.show()
        fig.savefig(f"Plot_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pdf")
        
        final_losses_train = []
        minimum_losses_test = []
        loss_values_test_tensor = torch.Tensor(loss_values_test)
        
        final_train_loss = loss_values_train[-1] #last value of training
        minimum_test_loss = torch.min(loss_values_test_tensor) #minimum value of training
        
        final_losses_train.append(final_train_loss)
        minimum_losses_test.append(minimum_test_loss)
        
        torch.save(final_losses_train,f"Final_losses_train_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
        torch.save(minimum_losses_test,f"Minimum_losses_test_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
        torch.save(loss_values_train, f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
    batch_size += 25
    #width += 10