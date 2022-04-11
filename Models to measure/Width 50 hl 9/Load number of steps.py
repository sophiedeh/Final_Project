# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:24:01 2022

@author: 20182672
"""
import torch
import matplotlib.pyplot as plt

batch_size = 100
dataset_size = 1000
epochs = 50000
training_times = 4
version = 7

width = 50
hidlay = 8

escape_steps = []
loss_value_difference = []
batch_sizes = []
es_ld_bs = [] # vector containing escape steps initial peak, loss depth initial peak and batch size

for u in range(version):
    for i in range(training_times):
        steps = torch.load(f"Steps_per_plateau_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
        escape_steps.appnd(steps[0])
        
        loss_value = torch.load(f"Filtered_loss_values_peaks_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
        loss_value_difference.append(loss_value[1]-loss_value[0])
        
        batch_sizes.append(batch_size)
        batch_size += 50
        
    es_ld_bs = list(zip(escape_steps, loss_value_difference, batch_sizes))
    torch.save(es_ld_bs,f"Escape steps and loss depth bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
              
    fig = plt.figure()
    fig.suptitle(f"Escape steps w{width} hl{hidlay} v{u+1}")
    fig.plot(batch_sizes, escape_steps,'-')
    plt.show()
    
    fig.savefig(f"Escape steps bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pdf")
    
    fig = plt.figure()
    fig.suptitle(f"Loss value depth w{width} hl{hidlay} v{u+1}")
    fig.plot(batch_sizes, escape_steps,'-')
    plt.show()
    
    fig.savefig(f"Loss value depth bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pdf")
    
    es_ld_bs = []
    escape_steps = []
    loss_value_difference = []
    batch_sizes = []
    
    