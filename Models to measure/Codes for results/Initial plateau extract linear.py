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
            number_of_steps_in_plateau = []
            total_steps_in_plateau = 0
            derivatives = []
            loss_values_downsampled = []
            
            # Load loss values
            loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt12.pt")
            
            #  Load the filtered training loss values and the derivatives from the trained networks.    
            # if batch_size < 225:
            #     loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt8.pt")
            # else: 
            #     loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt5.pt")
                    
            for i in range(filter_times):              
                  loss_values_train = savgol_filter(loss_values_train, window, 3) 
                        
            # Define number of steps and derivatives
            previous_loss = 0
            previous_steps = 0
            for i in range(len(loss_values_train)):
                number_of_steps.append((i+11)*(dataset_size/batch_size)) # i + 1 since first value corresponds to first epoch so not zero epoch
                der = (loss_values_train[i] - previous_loss)/(number_of_steps[i] - previous_steps)
                previous_loss = loss_values_train[i]
                previous_steps = number_of_steps[i]
                derivatives.append(der)
                    
            # Alter to torch tensors and absolute values for derivatives 
            derivatives_tensor = torch.DoubleTensor(derivatives)
            #derivatives_tensor_abs = torch.abs(derivatives_tensor)
            
            # Extract initial plateau
            derivatives_tensor = derivatives_tensor[100:1000]
            loss_values_train = loss_values_train[100:1000]
            number_of_steps = number_of_steps[100:1000]
            
          
            fig = plt.figure()
            ax = fig.subplots()
            fig.suptitle(f"Initial plateau derivatives bs{batch_size} w{width} hl{hidlay} v{u+1} ft{filter_times} win{window}")
            ax.plot(number_of_steps, derivatives_tensor,'-')
            ax.legend()
            ax.set_xlabel("Number of steps (-)")
            ax.set_ylabel("Derivatives (-)")
            plt.show()
            
            fig.savefig(f"Initial_plateau_derivatives_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pdf")
            
            fig = plt.figure()
            ax = fig.subplots()
            fig.suptitle(f"Initial plateau loss value plot bs{batch_size} w{width} hl{hidlay} v{u+1} ft{filter_times} win{window}")
            ax.plot(number_of_steps, loss_values_train,'-')
            ax.legend()
            ax.set_xlabel("Number of steps (-)")
            ax.set_ylabel("Loss value plot (-)")
            plt.show()
            
            fig.savefig(f"Initial_plateau_loss_value_plot_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pdf")
            
            batch_size += 25
            #hidlay += 1
    # width += 25
    # hidlay = 8
    #batch_size += 50 