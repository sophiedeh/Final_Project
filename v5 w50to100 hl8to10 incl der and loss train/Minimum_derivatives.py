# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 10:05:17 2022

@author: 20182672
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

batch_size = 200
dataset_size = 1000
epochs = 50000
training_times = 9

width = 75
hidlay = 9
different_depth = 1
different_width = 1
minimum_values_deriatives = []
number_of_steps = []
number_of_steps_in_plateau = []
total_steps_in_plateau = 0
derivatives = []

for i in range(different_width): 
    for i in range(different_depth):
            #  Load the training loss values and the derivatives from the trained networks.    
            #derivatives = torch.load(f"Derivatives_train_bs{batch_size}_w{width}_hl{hidlay}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
            loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
            
            loss_values_train = savgol_filter(loss_values_train, 51, 3)
            # Smoothen the loss values
            
            # Define number of steps and derivatives
            for i in range(len(loss_values_train)):
                number_of_steps.append((i+1)*(dataset_size/batch_size)) # i + 1 since first value corresponds to first epoch so not zero epoch
                np_loss_values_train = loss_values_train.detach().numpy()
                np_number_of_steps = number_of_steps.detach().numpy()
                der = np.diff(np_loss_values_train,i)/np.diff(np_number_of_steps,i)
                derivatives.append(der)
            
            # Alter to torch tensors and absolute values for derivatives 
            derivatives_tensor = torch.Tensor(derivatives)
            derivatives_tensor_abs = torch.abs(derivatives_tensor)
            loss_values_train_tensor = torch.Tensor(loss_values_train)
            
            # Find minimum value of derivative for each network, but seems to give to high result.
            minimum_derivatives = torch.min(derivatives_tensor) 
            
            for i in range(len(derivatives_tensor_abs)):
                if derivatives_tensor_abs[i+1] - derivatives_tensor_abs[i] < 10**(-1):
                    steps_in_plateau = number_of_steps[i+1] - number_of_steps[i]
                    total_steps_in_plateau += steps_in_plateau
                else:
                    number_of_steps_in_plateau.append(total_steps_in_plateau)
                    
            steps_per_plateau = number_of_steps_in_plateau[number_of_steps_in_plateau>0]
            torch.save(steps_per_plateau, f"Steps_per_plateau_bs{batch_size}_w{width}_hl{hidlay}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
           
            
            #minimum_values_deriatives.append(der)
            hidlay += 1
    width += 25
    hidlay = 8
    
steps_per_plateau = number_of_steps_in_plateau[number_of_steps_in_plateau>0]  
print('Steps per plateau')  

plt.plot(number_of_steps, derivatives_tensor_abs,'-')
plt.yscale('log')
