# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 14:52:27 2022

@author: 20182672
"""
def max_nodes_model(input, output, width, layers):
    total = input*width + layers*width*width + width*output
    return total

# N is number of parameters
# P is the to fitting datapoints

N = max_nodes_model(10, 1, 5, 5)
NP = N/1000

print(f"N = {N}")
print(f"N/P = {NP}")