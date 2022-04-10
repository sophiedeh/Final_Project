# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:22:52 2022

@author: 20182672
"""
import torch
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

batch_size = 200
dataset_size = 1000
epochs = 50000
training_times = 9

width = 75
hidlay = 9
different_depth = 1
different_width = 1
number_of_steps = []

#  Load the training loss values and the derivatives from the trained networks.    
loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")

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
fig.suptitle(f"Train vs Test Losses width{width} hidlay{hidlay}")
axs[0].plot(number_of_steps, loss_values_train_tensor,'-')
axs[0].set_title('Train_log')
axs[0].set_yscale('log')
axs[1].plot(number_of_steps, loss_values_train_tensor,'-')
axs[1].set_title('Train_linear')
