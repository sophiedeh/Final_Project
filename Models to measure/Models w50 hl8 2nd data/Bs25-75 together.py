# -*- coding: utf-8 -*-
"""
Created on Mon May 16 16:24:31 2022

@author: 20182672
"""

import torch
import matplotlib.pyplot as plt

width = 50
hidlay = 8
training_times = 12
dataset_size = 1000
version = 4
epochs = 50000


for u in range(version):
    batch_size = 25
    fig = plt.figure()
    plt.title(f"Loss values of network with width {width}, hidden layers {hidlay} and version {u+1}")
    for i in range(3):
        number_of_steps = []
        
        # Load loss values
        loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
        
        for i in range(len(loss_values_train)):
            number_of_steps.append((i+1)*(dataset_size/batch_size))
        
        #fig = plt.figure()
        #ax = fig.subplots()
        
        plt.plot(number_of_steps[0:10000], loss_values_train[0:10000],'-')
        plt.yscale('log')
        #plt.legend(f"{batch_size}")
        plt.xlabel("Number of steps (-)")
        plt.ylabel("Loss value (-)")
        
        batch_size += 25
    #fig = plt.figure()
    plt.legend(['Bs 25','Bs 50','Bs 75'])
    plt.show()
    fig.savefig(f"Batch size 25 - 75 loss values of w{width} hl{hidlay} v{u+1}.pdf", bbox_inches='tight')
    