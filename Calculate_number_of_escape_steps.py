# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 10:05:17 2022

@author: 20182672
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import pandas as pd

batch_size = 200
dataset_size = 1000
epochs = 50000
training_times = 9

width = 50
hidlay = 8
different_depth = 3
different_width = 3
#minimum_values_deriatives = []
number_of_steps = []
number_of_steps_in_plateau = []
total_steps_in_plateau = 0
derivatives = []

for i in range(different_width): 
    for i in range(different_depth):
            number_of_steps = []
            number_of_steps_in_plateau = []
            total_steps_in_plateau = 0
            derivatives = []
            
            #  Load the training loss values and the derivatives from the trained networks.    
            #derivatives = torch.load(f"Derivatives_train_bs{batch_size}_w{width}_hl{hidlay}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
            loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
            
            # for i in range(len(loss_values_train)):
            #     number_of_steps.append((i+1)*(dataset_size/batch_size))
            
            # window = 100
            # summed = 0
            # for i in range(len(loss_values_train)):
            #     for i in range(window):
            #         summed += loss_values_train[i] 
            #     loss_values_train[i] = summed/window
            
            # df = pd.DataFrame(loss_values_train)
            # Smooth loss values and create tensor
            filter_times = 1000
            for i in range(filter_times):
            #     #df.rolling(window=10).mean()
                 loss_values_train = savgol_filter(loss_values_train, 51, 3)
            loss_values_train_tensor = torch.Tensor(loss_values_train)
            
            # Define number of steps and derivatives
            previous_loss = 0
            previous_steps = 0
            for i in range(len(loss_values_train)):
                number_of_steps.append((i+1)*(dataset_size/batch_size)) # i + 1 since first value corresponds to first epoch so not zero epoch
                der = (loss_values_train[i] - previous_loss)/(number_of_steps[i] - previous_steps)
                previous_loss = loss_values_train[i]
                previous_steps = number_of_steps[i]
                derivatives.append(der)
                
                # np_loss_values_train = loss_values_train_tensor.detach().numpy()
                # np_number_of_steps = torch.Tensor(number_of_steps).detach().numpy()
                # der = np.diff(np_loss_values_train,i)/np.diff(np_number_of_steps,i)
                # derivatives.append(der)
            
            # Alter to torch tensors and absolute values for derivatives 
            derivatives_tensor = torch.Tensor(derivatives)
            derivatives_tensor_abs = torch.abs(derivatives_tensor)
            
            # Find peaks in derivatives
            peaks = find_peaks(derivatives_tensor_abs, height = 10**(-6), distance = 1000) #5000 geeft de twee meest zichtbare plateaus
            height = peaks[1]['peak_heights'] #list containing the height of the peaks
            peak_pos = (peaks[0]+1)*(dataset_size/batch_size)
            
            # Find minimum value of derivative for each network, but seems to give to high result.
            # minimum_derivatives = torch.min(derivatives_tensor) 
            
            fig = plt.figure()
            ax = fig.subplots()
            ax.plot(number_of_steps, derivatives_tensor_abs,'-')
            ax.set_yscale('log')
            ax.scatter(peak_pos, height, color = 'r', s = 10, marker = 'D', label = 'maxima')
            ax.legend()
            ax.grid()
            plt.show()
            
            fig.savefig(f"Derivatives_bs{batch_size}_w{width}_hl{hidlay}_ds{dataset_size}_e{epochs}_tt{training_times}.pdf")
            
            # for i in range(len(derivatives_tensor_abs)-1):
            #     if  derivatives_tensor_abs[i]/derivatives_tensor_abs[i+1] <= 10:
            #         #derivatives_tensor_abs[i]<10**(-7):
            #         steps_in_plateau = number_of_steps[i+1] - number_of_steps[i]
            #         total_steps_in_plateau += steps_in_plateau
            #     else:
            #         number_of_steps_in_plateau.append(total_steps_in_plateau)
            #         total_steps_in_plateau = 0
                    #  for filter 1000 Steps per plateau: tensor([66200.,   215.,  4465.,   615., 37035.])
                    
            for i in range(len(peak_pos)):
                if i == 0:
                    steps_in_plateau = peak_pos[i]
                    number_of_steps_in_plateau.append(steps_in_plateau)
                else:
                    steps_in_plateau = peak_pos[i]-peak_pos[i-1]
                    number_of_steps_in_plateau.append(steps_in_plateau)
                         
            number_of_steps_in_plateau_tensor = torch.Tensor(number_of_steps_in_plateau)
            number_of_steps_in_plateau_tensor = number_of_steps_in_plateau_tensor[number_of_steps_in_plateau_tensor>0]
            torch.save(number_of_steps_in_plateau_tensor, f"Steps_per_plateau_bs{batch_size}_w{width}_hl{hidlay}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
           
            #minimum_values_deriatives.append(der)
            hidlay += 1
    width += 25
    hidlay = 8
      
print('Steps per plateau:', number_of_steps_in_plateau_tensor)  

# plt.plot(number_of_steps, derivatives_tensor_abs,'-')
# plt.yscale('log')
