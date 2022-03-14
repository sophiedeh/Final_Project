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
  
d0 = pd.read_csv('train.csv')

# save the labels into a variable labels.
labels = d0['label']

# Drop the label feature and store the pixel data in data.
data = d0.drop("label", axis=1)

# Remove mean and scale to unit variance
standardized_data= StandardScaler().fit_transform(data)

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
mini_train = output_df.loc[:1999, cols[:11]]
mini_test = output_df.loc[2000:3999, cols[:11]]

mini_test.index = range(mini_test.shape[0])

print('mini_train.shape = ', mini_train.shape)
print('mini_test.shape = ', mini_test.shape)

mini_train.to_csv('mini_pca_train_2000.csv')
mini_test.to_csv('mini_pca_test_2000.csv')

# Also convert these into a pytorch friendly form
pt_train = []
pt_train_l = torch.zeros(mini_train.shape[0], 10)
cols = mini_train.columns

for row in range(mini_train.shape[0]):
    label = mini_train.loc[row, cols[0]]
    data_row = mini_train.loc[row, cols[1:]]
    data_row = np.expand_dims(data_row, axis=0)
    data_row = np.expand_dims(data_row, axis=0)
    pt_train_l[row, int(label)] = 1
    pt_train.append(torch.Tensor(data_row))

pt_train_tensor = torch.cat(pt_train)
pt_train_data = TensorDataset(pt_train_tensor, pt_train_l)

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
fname = 'mini_pca_train_2000.pt'
torch.save(pt_train_data, fname)

fname = 'mini_pca_test_2000.pt'
torch.save(pt_test_data, fname)










