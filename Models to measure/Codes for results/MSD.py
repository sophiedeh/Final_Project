# -*- coding: utf-8 -*-
"""
Created on Fri May 13 10:27:15 2022

@author: 20182672
"""

import torch
from Models_w50to100_hl8to10 import NeuralNetwork

training_times = 12
width = 5
hidlay = 5
epochs = 50000
version = 4
dataset_size = 1000
epoch = 0
batch_size = 25 
weights = []
initial_weights = []
MSD = []

for i in range(training_times): 
    for u in range(version):
        for t in range(epochs/100): 
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            # print(f"Using {device} device")
            
            # model = NeuralNetwork(width=width,hidlay=hidlay).to(device)
            # model.load_state_dict(torch.load(f"model_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{t}_tt{training_times}.pth", map_location=torch.device(device)))  
            
            model = NeuralNetwork(width=width,hidlay=hidlay)
            model.load_state_dict(torch.load(f"model_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epoch}_tt{training_times}.pth")) 
    
    
            if epoch == 100:
                initial_weights.append()
            else:
                weights.append()
            
            initial_weights = torch.Tensor(initial_weights)
            weights = torch.Tensor(weights)
            
            sub = torch.sub(initial_weights,weights) 
            squared = torch.square(sub)
            summed = torch.sum(squared)
            MSDi = (1/len(initial_weights)) * summed
            
            MSD.append(MSDi)
            epoch+=100
        torch.save(MSD,f"MSD_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}.pth")
        epoch = 0
    batch_size += 25