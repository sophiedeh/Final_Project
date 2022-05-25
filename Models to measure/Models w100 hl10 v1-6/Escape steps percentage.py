# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:15:11 2022

@author: 20182672
"""
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

dataset_size = 1000
epochs = 50000
training_times = 12

width = 100
hidlay = 10
version = 4

for u in range(version):
    batch_size = 25
    plt.figure()
    plt.title(f"Loss values of network with width {width}, hidden layers {hidlay} and version {u+1}")
    escape_steps_values = []
    escape_positions = []
    batch_sizes = []
    for i in range(training_times):
        number_of_steps = []
        
        # Load loss values
        loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
        loss_values = []
        
        for i in range(0, 50000, 100):
            loss_values.append(loss_values_train[i])
            number_of_steps.append((i+1)*(dataset_size/batch_size))
        # for i in range(len(loss_values_train)):
        #     number_of_steps.append((i+1)*(dataset_size/batch_size))
        
        interp = interp1d(number_of_steps, loss_values)
        fine_number_of_steps = np.arange(number_of_steps[0], max(number_of_steps), max(number_of_steps)/10000)
        fine_losses = interp(fine_number_of_steps)
        fine_losses = torch.from_numpy(fine_losses)
        
        fine_number_of_steps = torch.from_numpy(fine_number_of_steps)
        #fig = plt.figure()
        #ax = fig.subplots()
        	
        escapes_for_value = []
        for x in range(len(fine_losses)):
            escapes_for_value.append((x+1)*(dataset_size/batch_size))
                
        t = 1
        while fine_losses[t] >= 0.10*fine_losses[0]:
            t += 1
            escape_step_value = fine_losses[t+1]
            escape_pos = fine_number_of_steps[t+1]
        escape_steps_values.append(escape_step_value)
        escape_positions.append(escape_pos)
        
        plt.plot(fine_number_of_steps, fine_losses,'-')
        plt.yscale('log')
        #plt.legend(f"{batch_size}")
        plt.xlabel("Number of steps (-)")
        plt.ylabel("Loss value (-)")
        plt.scatter(escape_pos, escape_step_value,c='r')
        plt.xlim([80000,160000])
        plt.ylim(bottom=((10)**(-3)),top=10**0)
        
        batch_sizes.append(batch_size)
        batch_size += 25   
    plt.legend(['Bs 25','Bs 50', 'Bs 75','Bs 100', 'Bs 125', 'Bs 150', 'Bs 175', 'Bs 200','Bs 225', 'Bs 250', 'Bs 275', 'Bs 300'])
    plt.savefig(f"All loss values of w{width} hl{hidlay} v{u+1}.pdf",bbox_inches='tight')
    plt.show()
    
    plt.plot(batch_sizes, escape_positions,'-', marker='o')
    plt.title(f"Escape steps percentage of w{width} hl{hidlay} v{u+1}")
    plt.savefig(f"Escape steps percentage of w{width} hl{hidlay} v{u+1}.pdf",bbox_inches='tight')
    plt.show()