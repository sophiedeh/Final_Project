# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 10:05:17 2022

@author: 20182672
"""
import torch

batch_size = 200
dataset_size = 1000
epochs = 50000
training_times = 9

width = 50
hidlay = 8
different_depth = 3
different_width = 3
minimum_values_deriatives = []
escape_times = []
for i in range(different_width): 
    for i in range(different_depth):
            derivatives = torch.load(f"Derivatives_train_bs{batch_size}_w{width}_hl{hidlay}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
            epoch_times = torch.load(f"Epoch_times_train_bs{batch_size}_w{width}_hl{hidlay}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
            derivatives_tensor = torch.Tensor(derivatives)
            minimum_derivatives = torch.min(derivatives_tensor)
            for i in range(len(derivatives_tensor)):
                if derivatives_tensor[i]==minimum_derivatives:
                    escape_times.append(epoch_times[i])
            minimum_values_deriatives.append(torch.min(derivatives_tensor))
            hidlay += 1
    width += 25
    hidlay = 8
    
print(minimum_values_deriatives)
print(escape_times)