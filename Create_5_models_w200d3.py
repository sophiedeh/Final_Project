# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 11:27:35 2022

@author: 20182672
"""
import torch
from torch import nn

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, width):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10, width), 
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            #m.bias.data.fill_(0.01)
    
width = 200
depth = 3 
numbers_of_models = 5

for i in range(numbers_of_models): 
    model = NeuralNetwork(width=width)
    model.apply(init_weights)
    torch.save(model.state_dict(), f"initial_model_xavier_{i+1}_w{width}_d{depth}.pth")
    print(f"Saved PyTorch Model State to initial_model_xavier_{i+1}_w{width}_d{depth}.pth")