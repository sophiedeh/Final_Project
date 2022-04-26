# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 10:42:43 2022

@author: 20182672
"""

import torch
import matplotlib.pyplot as plt

batch_size = 200
dataset_size = 1000
epochs = 50000
training_times = 8
u = 0
width = 50
hidlay = 9

# Load loss values
loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")

fig = plt.figure()
ax = fig.subplots()
fig.suptitle(f"Loss plot model with batch size {batch_size}, width {width} and hidden layers {hidlay}")
ax.plot(loss_values_train,'-')
ax.set_yscale('log')
ax.grid()
plt.show()

fig.savefig(f"Loss value of trained model with batch size {batch_size}, width {width} and hidden layers {hidlay}.pdf")