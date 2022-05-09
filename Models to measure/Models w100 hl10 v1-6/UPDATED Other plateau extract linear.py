# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 08:03:29 2022

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

batch_size = 100
u = 0
number_of_steps = []

# Load loss values
loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")

#  Load the filtered training loss values and the derivatives from the trained networks.    
# if batch_size < 225:
#     loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt8.pt")
# else: 
#     loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt5.pt")
    
for i in range(filter_times):              
  loss_values_train_filtered = savgol_filter(loss_values_train, window, 3) 
        
# Define number of steps
for i in range(len(loss_values_train)):
    number_of_steps.append((i+1)*(dataset_size/batch_size)) # i + 1 since first value corresponds to first epoch so not zero epoch
    
# Extract initial plateau
loss_values_train = loss_values_train[13200:13300]
loss_values_train_filtered = loss_values_train_filtered[13200:13300]
number_of_steps = number_of_steps[13200:13300]

fig = plt.figure()
ax = fig.subplots()
fig.suptitle(f"Second and more plateaus loss value bs{batch_size} w{width} hl{hidlay} v{u+1} ft{filter_times} win{window}")
ax.plot(number_of_steps, loss_values_train,'-')
ax.legend()
ax.set_xlabel("Number of steps (-)")
ax.set_ylabel("Loss value (-)")
plt.show()

fig.savefig(f"Other_plateau_loss_value_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pdf")

fig = plt.figure()
ax = fig.subplots()
fig.suptitle(f"Second and more plateaus loss value filtered bs{batch_size} w{width} hl{hidlay} v{u+1} ft{filter_times} win{window}")
ax.plot(number_of_steps, loss_values_train,'-')
ax.legend()
ax.set_xlabel("Number of steps (-)")
ax.set_ylabel("Loss value (-)")
plt.show()

fig.savefig(f"Other_plateau_loss_value_filtered_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pdf")


