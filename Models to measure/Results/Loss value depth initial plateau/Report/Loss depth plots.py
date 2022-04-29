# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 09:40:28 2022

@author: 20182672
"""
import torch
import matplotlib.pyplot as plt

batch_size = 25
dataset_size = 1000
epochs = 50000
training_times = 12
version = 4

width = 100
hidlay = 10

escape_steps = []
loss_value_difference = []
batch_sizes = []
es_ld_bs = [] # vector containing escape steps initial peak, loss depth initial peak and batch size

for u in range(version):
    for i in range(training_times):
        #steps = torch.load(f"ABS_Steps_per_plateau_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
        #escape_steps.append(steps[0])
        
        loss_value = torch.load(f"ABS_Filtered_loss_values_peaks_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
        loss_value_difference.append(loss_value[1]-loss_value[0])
                  
        batch_sizes.append(batch_size)
        batch_size += 25
    batch_size = 25
    
    # es_ld_bs = list(zip(escape_steps, loss_value_difference, batch_sizes))
    # torch.save(es_ld_bs,f"Escape steps and loss depth bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
    
    #es_bs = list(zip(escape_steps, batch_sizes))
    #torch.save(es_bs,f"Escape steps bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
    
    es_ld = list(zip(loss_value_difference, batch_sizes))
    torch.save(es_ld,f"Loss value depth bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
              
    # fig = plt.figure()
    # ax = fig.subplots()
    # fig.suptitle(f"Escape steps w{width} hl{hidlay} v{u+1}")
    # ax.plot(batch_sizes, escape_steps,'-')
    # ax.set_xlabel("Number of steps (-)")
    # ax.set_ylabel("Absolute derivative (-)")
    # plt.scatter(batch_sizes, escape_steps)
    # plt.show()
    
    # fig.savefig(f"Escape steps w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pdf")
    
    fig = plt.figure()
    ax = fig.subplots()
    fig.suptitle(f"Loss value depth versus batch size")
    ax.set_xlabel("Batch size (-)")
    ax.set_ylabel("Loss value depth (-)")
    ax.plot(batch_sizes, loss_value_difference,'-')
    plt.show()
    
    fig.savefig(f"Loss value depth w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.png",bbox_inches="tight")
    
    es_ld_bs = []
    es_bs = []
    es_ld = []
    escape_steps = []
    loss_value_difference = []
    batch_sizes = []
    