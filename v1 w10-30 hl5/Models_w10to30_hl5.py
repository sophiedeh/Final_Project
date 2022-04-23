# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:12:26 2022

@author: 20182672
"""
import torch
from torch import nn

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, width, hidlay):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        initial_layer = nn.Linear(10, width)
        hidden_layers = []
        for i in range(hidlay):
            hidden_layers.append(nn.Linear(width,width))
            hidden_layers.append(nn.ReLU())
        final_layer = nn.Linear(width, 1)
        self.linear_relu_stack = nn.Sequential(
            initial_layer,
            *hidden_layers,
            final_layer
        )
   
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
width = 10
hidlay = 5
different_depth = 1
different_width = 3
version = 4
#for i in range(different_depth): 
for i in range(different_width):
    for u in range(version):
        model = NeuralNetwork(width=width, hidlay=hidlay)
        torch.save(model.state_dict(), f"initial_model_w{width}_hl{hidlay}_v{u+1}.pth")
        print(f"Saved PyTorch Model State to initial_model_w{width}_hl{hidlay}.pth")
    width += 10
    #hidlay += 1
   

