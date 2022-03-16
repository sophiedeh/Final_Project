#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:31:59 2022

@author: mwinter
"""

# A script to produce a binary classification version of MNIST, where the 
# digits are sorted into two classes: odd and even numbers.


import torch
from torch import nn
from torch.utils.data import TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os

### IF DEBUG IS TRUE NO DATA IS SAVED ###
debug = False

current_dir = os.getcwd()
if current_dir[:14] == '/Users/mwinter':
    base_path = '/Users/mwinter/Documents/python_code/NN_measurements/'
else:
    base_path = '/home/phys/20214003/python_code/NN_measurements/'
   
data_input_dir = '{}data/downsampled_MNIST/'.format(base_path)
        
fname = data_input_dir + 'training_data.pt'
training_data = torch.load(fname)

bm_train = []
bm_train_l = torch.zeros(len(training_data), 1)

for count, my_tuple in enumerate(training_data):
    img = my_tuple[0]
    label = my_tuple[1]
    
    if label%2==0:
        bm_train_l[count] = 1
    else:
        bm_train_l[count] = -1
        
    bm_train.append(img)

bm_train_tensor = torch.cat(bm_train)
bm_train_data = TensorDataset(bm_train_tensor, bm_train_l)


# Download test data from open datasets.
fname = data_input_dir + 'test_data.pt'
test_data = torch.load(fname)

bm_test = []
bm_test_l = torch.zeros(len(test_data), 1)

for count, my_tuple in enumerate(test_data):
    img = my_tuple[0]
    label = my_tuple[1]
    
    if label%2==0:
        bm_test_l[count] = 1
    else:
        bm_test_l[count] = -1
        
    bm_test.append(img)

bm_test_tensor = torch.cat(bm_test)
bm_test_data = TensorDataset(bm_test_tensor, bm_test_l)


if not debug:
    # Save pytorch data
    data_output_dir = '{}data/binary_MNIST/'.format(base_path)
    dir_exists = os.path.isdir(data_output_dir)
    if not dir_exists:
        os.mkdir(data_output_dir)
    
    fname = data_output_dir + 'binary_MNIST_train.pt'
    torch.save(bm_train_data, fname)
    
    fname = data_output_dir + 'binary_MNIST_test.pt'
    torch.save(bm_test_data, fname)

