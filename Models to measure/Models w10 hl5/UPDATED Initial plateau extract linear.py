# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 10:05:17 2022

@author: 20182672
"""
import torch
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, sosfiltfilt, butter

dataset_size = 1000
epochs = 50000
training_times = 12 
# Set training time to 1 to determine per batch size what an appropriate length is

width = 10
hidlay = 5
filter_times = 100
window = 101
version = 4

for u in range(version):
    batch_size = 25
    set_range = 1000 # every batch size step 1000 more
    for i in range(training_times):
            number_of_steps = []
                       
            # Load loss values
            loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt12.pt")
            
            # Load the filtered training loss values and the derivatives from the trained networks.    
            # if batch_size < 225:
            #     loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt8.pt")
            # else: 
            #     loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt5.pt")
                    
            for i in range(filter_times):              
                   loss_values_train_filtered = savgol_filter(loss_values_train, window, 3) 
                        
            # Define number of steps
            for i in range(len(loss_values_train)):
                number_of_steps.append((i+1)*(dataset_size/batch_size)) # i + 1 since first value corresponds to first epoch so not zero epoch        
            
            # Extract initial plateau
            loss_values_train = loss_values_train[100:set_range]
            loss_values_train_filtered = loss_values_train_filtered[100:set_range]
            number_of_steps = number_of_steps[100:set_range] 
          
            fig = plt.figure()
            ax = fig.subplots()
            fig.suptitle(f"Initial plateau filtered loss bs{batch_size} w{width} hl{hidlay} v{u+1} ft{filter_times} win{window}")
            ax.plot(number_of_steps, loss_values_train_filtered,'-')
            ax.legend()
            ax.set_xlabel("Number of steps (-)")
            ax.set_ylabel("Filtered loss value (-)")
            plt.show()
            
            fig.savefig(f"Initial_plateau_filtered_loss_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pdf")
            
            fig = plt.figure()
            ax = fig.subplots()
            fig.suptitle(f"Initial plateau original loss value plot bs{batch_size} w{width} hl{hidlay} v{u+1}")
            ax.plot(number_of_steps, loss_values_train,'-')
            ax.legend()
            ax.set_xlabel("Number of steps (-)")
            ax.set_ylabel("Loss value (-)")
            plt.show()
            
            fig.savefig(f"Initial_plateau_loss_value_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pdf")
            
            batch_size += 25
            set_range += 1000