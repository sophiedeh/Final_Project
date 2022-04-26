# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:47:57 2022

@author: 20182672
"""
import torch
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, sosfiltfilt, butter

dataset_size = 1000
epochs = 50000
training_times = 12

width = 100
hidlay = 10
different_depth = 1
different_width = 1
filter_times = 100
window = 101
version = 4

for u in range(version):
    batch_size = 25
    for i in range(training_times):
        number_of_steps = []
        
        # Load loss values
        loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
        
        for i in range(filter_times):              
              loss_values_train = savgol_filter(loss_values_train, window, 3)
              
        for i in range(len(loss_values_train)):
            number_of_steps.append((i+11)*(dataset_size/batch_size)) # i + 1 since first value corresponds to first epoch so not zero epoch
            
        fig = plt.figure()
        ax = fig.subplots()
        fig.suptitle(f"Filtered loss plot batch size {batch_size} width {width} hidden layers {hidlay} version {u+1}")
        ax.plot(number_of_steps, loss_values_train,'-')
        ax.set_yscale('log')
        ax.grid()
        plt.show()
        
        fig.savefig(f"Filtered loss plot_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pdf")