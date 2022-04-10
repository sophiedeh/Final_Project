# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:35:32 2022

@author: 20182672
"""

import torch
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
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
#derivatives = torch.load(f"Derivatives_train_bs{batch_size}_w{width}_hl{hidlay}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
#loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")

window = 2
a = torch.rand(10)
print(f"First time:{a}")

df = pd.DataFrame(a)
# Smooth loss values and create tensor

# df[0].plot()
# df[0].rolling(window=window).mean().plot()

# filter_times = 1
# for i in range(filter_times):
#      df[2] = df[0].rolling(window=window).mean()
#      #loss_values_train = savgol_filter(loss_values_train, 51, 9)
# a = torch.Tensor(df[2])

a = df[0].mean()

# window = 2
# summed = 0
# for i in range(len(a)):
#     while i <= window:
#         summed += a[i] 
#         a[i] = summed/window
#     else:
#         break
     
print(f"Second time:{a}")