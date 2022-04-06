# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:24:01 2022

@author: 20182672
"""
import torch

batch_size = 200
dataset_size = 1000
epochs = 50000
training_times = 9

width = 50
hidlay = 8

steps = torch.load(f"Steps_per_plateau_bs{batch_size}_w{width}_hl{hidlay}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
print(steps)