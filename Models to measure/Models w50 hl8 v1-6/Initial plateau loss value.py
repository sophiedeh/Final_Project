# -*- coding: utf-8 -*-
"""
Created on Mon May  2 10:31:04 2022

@author: 20182672
"""
import torch
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, sosfiltfilt, butter

dataset_size = 1000
epochs = 50000
training_times = 1

width = 50
hidlay = 8
different_depth = 1
different_width = 1
filter_times = 100
window = 101
version = 4

# number_of_steps = []
# number_of_steps_in_plateau = []
# loss_values_at_peaks = []
# total_steps_in_plateau = 0
# derivatives = []

for u in range(version):
    batch_size = 25
    for i in range(training_times):
            number_of_steps = []
            
            # Load loss values
            loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt12.pt")
            
            #  Load the filtered training loss values and the derivatives from the trained networks.    
            # if batch_size < 225:
            #     loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt8.pt")
            # else: 
            #     loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt5.pt")
                    
            for i in range(filter_times):              
                  loss_values_train = savgol_filter(loss_values_train, window, 3) 
                        
            # Define number of steps
            for i in range(len(loss_values_train)):
                number_of_steps.append((i+11)*(dataset_size/batch_size)) # i + 1 since first value corresponds to first epoch so not zero epoch
              
            # Extract initial plateau
            loss_values_train = loss_values_train[100:1000]
            number_of_steps = number_of_steps[100:1000]
            
            fig = plt.figure()
            ax = fig.subplots()
            fig.suptitle(f"Initial plateau loss value plot bs{batch_size} w{width} hl{hidlay} v{u+1} ft{filter_times} win{window}")
            ax.plot(number_of_steps, loss_values_train,'-')
            #ax.scatter(peak_pos, height, color = 'r', s = 10, marker = 'D', label = 'minima')
            ax.legend()
            ax.set_xlabel("Number of steps (-)")
            ax.set_ylabel("Loss value plot (-)")
            plt.show()
            
            fig.savefig(f"Initial_plateau_loss_value_plot_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pdf")
            
            #batch_size += 25
            #hidlay += 1
    # width += 25
    # hidlay = 8
    #batch_size += 50 