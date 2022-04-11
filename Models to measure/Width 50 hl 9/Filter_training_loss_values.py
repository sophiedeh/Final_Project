# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:22:52 2022

@author: 20182672
"""
import torch
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

batch_size = 100
dataset_size = 1000
epochs = 50000
training_times = 4
version = 1

width = 50
hidlay = 9
different_depth = 1
different_width = 1


for u in range(version): 
    for i in range(training_times):
        number_of_steps = []
        loss_values_downsampled = []
        
        #  Load the training loss values and the derivatives from the trained networks.    
        loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
        
        # for i in range(len(loss_values_train)):
        #     if i % 100 == 0:
        #         loss_values_downsampled.append(loss_values_train[i])
        #         number_of_steps.append((i+1)*(dataset_size/batch_size))
                
        # Define number of steps array
        for i in range(len(loss_values_train)):
              number_of_steps.append((i+1)*(dataset_size/batch_size))
        
        # Smooth loss values and create tensor
        filter_times = 100
        window = 101
        for i in range(filter_times):              
              loss_values_train = savgol_filter(loss_values_train, window, 3)
        loss_values_train_tensor = torch.Tensor(loss_values_train)
        
        fig, axs = plt.subplots(2,sharex=True)
        fig.suptitle(f"Train vs Test Losses width{width} hidlay{hidlay} bs{batch_size}")
        axs[0].plot(number_of_steps, loss_values_train,'-')
        axs[0].set_title('Train_log')
        axs[0].set_yscale('log')
        axs[1].plot(number_of_steps, loss_values_train,'-')
        axs[1].set_title('Train_linear')
        
        #torch.save(loss_values_train_tensor,f"Filtered_loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_ds{dataset_size}_e{epochs}_tt{training_times}_ft{filter_times}_win{window}.pt")
        batch_size += 50