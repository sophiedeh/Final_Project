# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 14:52:27 2022

@author: 20182672
"""
def max_nodes_model(input, output, width, layers):
    total = input*width + layers*width*width + width*output
    return total

P = max_nodes_model(10, 1, 20, 7)
NP = 1000/P

print(f"P = {P}")
print(f"N/P = {NP}")