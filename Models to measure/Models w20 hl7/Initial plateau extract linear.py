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

width = 20
hidlay = 7
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
            loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
            
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
            derivatives_tensor = derivatives_tensor[200:3000]
            number_of_steps = number_of_steps[200:3000]
            
            # Find peaks in derivatives
            # peaks = find_peaks(derivatives_tensor, height = 10**(-5), distance = 1000) #5000 geeft de twee meest zichtbare plateaus
            # height = peaks[1]['peak_heights'] #list containing the height of the peaks
            # peak_pos = (peaks[0]+1)*(dataset_size/batch_size)
            
            #Finding the minima
            # derivatives_tensor_minima = torch.mul(derivatives_tensor,-1)
            # peaks = find_peaks(derivatives_tensor_minima, height = 5*10**(-5), distance = 1000)
            # peak_pos = (peaks[0]+1)*(dataset_size/batch_size) #list of the minima positions
            # height = derivatives_tensor[peaks[0]] #list of the mirrored minima heights
            
            fig = plt.figure()
            ax = fig.subplots()
            fig.suptitle(f"Initial plateau derivatives bs{batch_size} w{width} hl{hidlay} v{u+1} ft{filter_times} win{window}")
            ax.plot(number_of_steps, derivatives_tensor,'-')
            #ax.scatter(peak_pos, height, color = 'r', s = 10, marker = 'D', label = 'minima')
            ax.legend()
            ax.set_xlabel("Number of steps (-)")
            ax.set_ylabel("Derivatives (-)")
            plt.show()
            
            fig.savefig(f"Initial_plateau_derivatives_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pdf")
            
            # loss_values_at_peaks = []
            # loss_values_at_peaks.append(loss_values_train[0])
            # for i in range(len(peak_pos)):
            #     loss_values_at_peaks.append(loss_values_train[peaks[0][i]])
            #     if i == 0:
            #         steps_in_plateau = peak_pos[i]
            #         number_of_steps_in_plateau.append(steps_in_plateau)
            #     else:
            #         steps_in_plateau = peak_pos[i]-peak_pos[i-1]
            #         number_of_steps_in_plateau.append(steps_in_plateau)
                         
            # number_of_steps_in_plateau_tensor = torch.Tensor(number_of_steps_in_plateau)
            # number_of_steps_in_plateau_tensor = number_of_steps_in_plateau_tensor[number_of_steps_in_plateau_tensor>0]
            # torch.save(number_of_steps_in_plateau, f"Steps_per_plateau_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
            # torch.save(loss_values_at_peaks,f"Filtered_loss_values_peaks_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
            
            batch_size += 25
            #hidlay += 1
    # width += 25
    # hidlay = 8
    #batch_size += 50 