# -*- coding: utf-8 -*-
"""
Created on Mon May 16 16:27:49 2022

@author: 20182672
"""

import torch
import matplotlib.pyplot as plt

width = 100
hidlay = 10
training_times = 12
dataset_size = 1000
version = 4
epochs = 50000
colls = 3 

for u in range(version):
    batch_size = 25
    
    
    plt.title(f"Loss values of network with width {width}, hidden layers {hidlay} and version {u+1}")
    row = 0
    col = 0
    for i in range(training_times):
        number_of_steps = []
        if row==4:
            row = 0
        
        if col==3:
            col = 0
            
        # Load loss values
        loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
        
        for i in range(len(loss_values_train)):
            number_of_steps.append((i+1)*(dataset_size/batch_size))
        
        fig, axs = plt.subplots(4,3)
        axs[row,col].plot(number_of_steps[0:10000], loss_values_train[0:10000],'-')
        axs[row,col].set_yscale('log')
        #plt.legend(f"{batch_size}")
        axs[row,col].set_xlabel("Number of steps (-)")
        axs[row,col].set_ylabel("Loss value (-)")
        
        batch_size += 25
        row += 1
        col += 1
    axs.legend(['Bs 25','Bs 50', 'Bs 75','Bs 100', 'Bs 125', 'Bs 150', 'Bs 175', 'Bs 200','Bs 225', 'Bs 250', 'Bs 275', 'Bs 300'])
    plt.show()
    plt.savefig(f"All loss values of w{width} hl{hidlay} v{u+1}.jpg")
