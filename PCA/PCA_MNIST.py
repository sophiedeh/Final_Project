# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 09:32:15 2022

@author: 20182672
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 11:21:04 2021

@author: mwinter
"""

# A script to produce a PCA representation of the MNIST dataset
# This script borrows heavily from 
# https://github.com/ranasingh-gkp/PCA-TSNE-on-MNIST-dataset


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import torch
from torch.utils.data import DataLoader, TensorDataset
import os

### IF DEBUG IS TRUE NO DATA IS SAVED ###
debug = False
  
d0 = pd.read_csv('train.csv')

# save the labels into a variable labels.
labels = d0['label']

# Drop the label feature and store the pixel data in data.
data = d0.drop("label", axis=1)

# Remove mean and scale to unit variance
standardized_data = StandardScaler().fit_transform(data)

# Do PCA
pca = decomposition.PCA()
pca.n_components = standardized_data.shape[1]
pca_data = pca.fit_transform(standardized_data)

print('Shape data = ', data.shape)
print('Shape pca_data = ', pca_data.shape)

labels = np.expand_dims(labels, axis=1)
output_data = np.concatenate((labels, pca_data), axis=1)
cols = d0.columns

output_df = pd.DataFrame(output_data, columns=cols)
output_df.to_csv('pca_train.csv')

# Make mini train and test sets with first 10 PCA components and 1000 data points
# Note pandas .loc slicing includes start AND END points, but cols[:11] does not
mini_train = output_df.loc[1000:1999, cols[:11]]
mini_test = output_df.loc[2000:2999, cols[:11]] #creates tensor (1000,11)

mini_test.index = range(mini_test.shape[0])

print('mini_train.shape = ', mini_train.shape)
print('mini_test.shape = ', mini_test.shape)

mini_train.to_csv('mini_pca_train.csv')
mini_test.to_csv('mini_pca_test.csv')

# Also convert these into a pytorch friendly form
pt_train = []
pt_train_l = torch.zeros(mini_train.shape[0], 10) #tensor with vectors where index can correspond to label
cols = mini_train.columns

for row in range(mini_train.shape[0]):
    label = mini_train.loc[row, cols[0]] 
    data_row = mini_train.loc[row, cols[1:]] #data of the images PCA components
    data_row = np.expand_dims(data_row, axis=0) #creates one vector 
    pt_train_l[row, int(label)] = 1 #tensor where a 1 is indexed for correct column with label
    pt_train.append(torch.Tensor(data_row))

pt_train_tensor = torch.cat(pt_train) #alter sequence of vectors to tensor
pt_train_data = TensorDataset(pt_train_tensor, pt_train_l) #dataset

pt_test = []
pt_test_l = torch.zeros(mini_test.shape[0], 10)
cols = mini_test.columns

for row in range(mini_test.shape[0]):
    label = mini_test.loc[row, cols[0]]
    data_row = mini_test.loc[row, cols[1:]]
    data_row = np.expand_dims(data_row, axis=0)
    data_row = np.expand_dims(data_row, axis=0)
    pt_test_l[row, int(label)] = 1
    pt_test.append(torch.Tensor(data_row))

pt_test_tensor = torch.cat(pt_test)
pt_test_data = TensorDataset(pt_test_tensor, pt_test_l)

# Save pytorch data
fname_train = 'mini_pca_train.pt'
torch.save(pt_train_data, fname_train)

fname_test = 'mini_pca_test.pt'
torch.save(pt_test_data, fname_test)

# Binary classification of train
training_data = torch.load(fname_train)

bm_train = []
bm_train_l = torch.zeros(len(training_data), 1) #one column

for count, my_tuple in enumerate(training_data):
    img = my_tuple[0] #array of 10 PCA
    img = torch.unsqueeze(img,0)
    label = my_tuple[1] #array of 10 index
    label_int = label.argmax(0)
    
    if label_int%2==0:
        bm_train_l[count] = 1
    else:
        bm_train_l[count] = -1
        
    bm_train.append(img)

bm_train_tensor = torch.cat(bm_train)  
bm_train_data = TensorDataset(bm_train_tensor, bm_train_l)

# Binary classification of test
test_data = torch.load(fname_test)

bm_test = []
bm_test_l = torch.zeros(len(test_data), 1)

for count, my_tuple in enumerate(test_data):
    img = my_tuple[0]
    img = torch.unsqueeze(img,0) 
    label = my_tuple[1]
    label_int = label.argmax()
    
    if label_int%2==0:
        bm_test_l[count] = 1
    else:
        bm_test_l[count] = -1
        
    bm_test.append(img)

bm_test_tensor = torch.cat(bm_test)
bm_test_data = TensorDataset(bm_test_tensor, bm_test_l)

if not debug:
    fname_train = 'binary_MNIST_pca_train_2nd_version.pt'
    torch.save(bm_train_data, fname_train)
    
    fname_test = 'binary_MNIST_pca_test_2nd_version.pt'
    torch.save(bm_test_data, fname_test)