# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:35:32 2022

@author: 20182672
"""

import torch
import pandas as pd

batch_size = 200
dataset_size = 1000
epochs = 50000
training_times = 9

width = 75
hidlay = 9
different_depth = 1
different_width = 1
number_of_steps = []

#  Load the training loss values and the derivatives from the trained networks.    
loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")

df = pd.DataFrame(loss_values_train)
# Smooth loss values and create tensor

filter_times = 1
window = 101
for i in range(filter_times):
    df[1] = df[0].rolling(window=window).mean()

     # Code for EWM
     # df[1] = df[0].ewm(alpha=.75).mean()
     
     # Code for CMA
     # df[1] = df[0].expanding(1).sum()
     # for i in range(len(df[1])):
     #     df[1][i] = df[1][i]/(i+1)                   
     #loss_values_train = savgol_filter(loss_values_train, 51, 9)
loss_values_train_tensor = torch.Tensor(df[1])