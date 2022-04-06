#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 13:00:14 2022

@author: mwinter
"""

# A script to plot the training loss from a folder of saved models

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import os

# Define model class
class NeuralNetwork(nn.Module):
    def __init__(self, n_in=28*28, n_out=10, h_layer_widths=[512], bias=False,
                 gaussian_prior=False):
        super(NeuralNetwork, self).__init__()

        self.flatten = nn.Flatten()

        layer_widths = [n_in] + h_layer_widths
        net_depth = len(layer_widths)+1

        layers = []
        for count in range(len(layer_widths)-1):
            layers.append(
                nn.Linear(int(layer_widths[count]), int(layer_widths[count+1]),
                          bias=bias)
                         )

            layers.append(nn.ReLU())

        layers.append(nn.Linear(int(layer_widths[-1]), n_out, bias=bias))

        self.net = nn.Sequential(*layers)
        
        if gaussian_prior:
            self.net.apply(self.init_weights)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.net(x)
        return logits
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.25)

# Load existing models
def load_models(input_dir, model_name, params, device):
    files = os.listdir(input_dir)
    
    # order files
    epochs = []
    for file in files:
        file2 = file
        try:
            start, end = file2.split('_epoch_')
        except ValueError:
            continue
        
        if start != '{}_model'.format(model_name):
            continue
        
        epoch = int(end[:-4])
        epochs.append(epoch)
    
    epochs.sort()
    
    # load models
    N_inputs = params['N_inputs']
    N_outputs = params['N_outputs']
    h_layer_widths = params['h_layer_widths']
    
    models = []
    for epoch in epochs:
        filename = '{}{}_model_epoch_{}.pth'.format(input_dir, model_name, 
                                                     epoch)
        model = NeuralNetwork(n_in=N_inputs, n_out=N_outputs, 
                              h_layer_widths=h_layer_widths).to(device)
        model.load_state_dict(torch.load(filename, 
                                         map_location=torch.device(device)))
        
        models.append(model)
    
    return models, epochs

def make_quadratic_hinge_loss():
    
    def quadratic_hinge(output, target):
        Delta = 1.0-target*output
        zeros = torch.zeros_like(Delta)
        
        max_Delta = torch.max(zeros, Delta)
        
        sq_max_Delta = max_Delta*max_Delta
        
        return 0.5*torch.mean(sq_max_Delta)
    
    return quadratic_hinge

# Load a dataset
def load_dataset(dataset, batch_size, base_path, model_name=None):
    # Download training and testing data from open datasets 
    if dataset=='MNIST':
        training_data = datasets.MNIST(root="data", train=True, download=True,
                                        transform=ToTensor())
        test_data = datasets.MNIST(root="data", train=False, download=True,
                                    transform=ToTensor())
    elif dataset=='downsampled_MNIST':
        data_output_dir = '{}data/downsampled_MNIST/'.format(base_path)
        
        fname = data_output_dir + 'training_data.pt'
        training_data = torch.load(fname)
        
        fname = data_output_dir + 'test_data.pt'
        test_data = torch.load(fname)
        
    elif dataset=='binary_MNIST':
        data_output_dir = '{}data/binary_MNIST/'.format(base_path)
        
        fname = data_output_dir + 'binary_MNIST_train.pt'
        training_data = torch.load(fname)
        
        fname = data_output_dir + 'binary_MNIST_test.pt'
        test_data = torch.load(fname)
        
    elif dataset=='binary_PCA_MNIST':
        data_output_dir = '{}data/binary_PCA_MNIST/'.format(base_path)
        
        fname = data_output_dir + 'binary_PCA_MNIST_train.pt'
        training_data = torch.load(fname)
        
        fname = data_output_dir + 'binary_PCA_MNIST_test.pt'
        test_data = torch.load(fname)
        
    elif dataset=='mini_pca_mnist':
        data_output_dir = '{}data/PCA_MNIST/'.format(base_path)
        
        fname = data_output_dir + 'mini_pca_train.pt'
        training_data = torch.load(fname)
        
        fname = data_output_dir + 'mini_pca_test.pt'
        test_data = torch.load(fname)
        
    elif dataset=='teacher':
        data_output_dir = '{}data/teacher/'.format(base_path)
        
        fname = data_output_dir + 'teacher_train.pt'
        training_data = torch.load(fname)
        
        fname = data_output_dir + 'teacher_test.pt'
        test_data = torch.load(fname)
        
    else:
        print('PROVIDE A DATASET')
    
    # Get img size. data element is (tensor, lbl), image is tensor with 
    # shape either [1, rows, cols], or [rows, cols]. PCA vector is tensor with 
    # shape [1, rows]
    if dataset in ['mini_pca_mnist', 'teacher']:
        sample_vec = training_data[0][0][0, :]
        N_inputs = sample_vec.shape[0]
    elif dataset in ['binary_MNIST', 'binary_PCA_MNIST']:
        sample_img = training_data[0][0]
        N_inputs = sample_img.shape[0]*sample_img.shape[1]
    else:
        sample_img = training_data[0][0][0, :, :]
        N_inputs = sample_img.shape[0]*sample_img.shape[1]
    
    if dataset in ['teacher', 'binary_MNIST', 'binary_PCA_MNIST']:
        N_outputs = len(training_data[0][1])
    
    else:
        unique_labels = []
        for t in training_data:
            try:
                if len(t[1])==1:
                    try:
                        label = t[1].item()
                    except AttributeError:
                        label = t[1]
                else:
                    label = np.argmax(t[1]).item()
            
            except TypeError:
                label = t[1]
                
            if label not in unique_labels:
                unique_labels.append(label)
        
        
        N_outputs = len(unique_labels)
    
    if model_name == 'teacher':
        N_outputs = 1
    
    # Deal with batch size = dataset size case
    set_b_for_test = False
    if batch_size==-1:
        batch_size = len(training_data)
        set_b_for_test = True
    
    train_dataloader = DataLoader(training_data, batch_size=batch_size, 
                                  shuffle=True, drop_last=True)
    
    # Deal with batch size = dataset size case
    if set_b_for_test:
        batch_size = len(test_data)
        
    test_dataloader = DataLoader(test_data, batch_size=batch_size, 
                                 shuffle=True, drop_last=True)
    
    return (train_dataloader, test_dataloader, training_data, test_data, 
            N_inputs, N_outputs)

# Calculate loss on training dataset
def evaluate_loss(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    loss /= num_batches
    correct /= size
    return loss

# Plot evolution of training loss as model is trained
def plot_training_loss(models, epochs, model_name, params, device, base_path):
    batch_size = params['batch_size']
    dataset = params['dataset']
    
    (train_dataloader, test_dataloader, training_data, test_data, 
        N_inputs, N_outputs) = load_dataset(dataset, batch_size, base_path)
        
    # Define loss function 
    loss_function = params['loss_function']
    if loss_function == 'CrossEntropy':
        loss_fn = nn.CrossEntropyLoss()
        
    elif loss_function == 'MSELoss':
        loss_fn = nn.MSELoss()
        
    elif loss_function == 'Hinge':
        loss_fn = nn.HingeEmbeddingLoss()
        
    elif loss_function == 'quadratic_hinge':
        loss_fn = make_quadratic_hinge_loss()
        
    else:
        print('PROVIDE A LOSS FUNCTION')
    
    loss = []
    subsampled_epochs = []
    
    if len(models)>900:
        rate = 10
        print('Subsampling models to speed up training loss plotting')
    else:
        rate = 1
        print('Rate set to 1.')
    
    for count in range(0, len(epochs), rate):
        epoch = epochs[count]
        model = models[count]
        
        print('Calculating training loss at epoch {}'.format(epoch))

        l = evaluate_loss(train_dataloader, model, loss_fn, device)
        
        loss.append(l)
        subsampled_epochs.append(epoch)
    
    # Rescale time to be in time steps not epochs
    sub_epoch_array = np.array(subsampled_epochs)
    N_steps = len(train_dataloader)
    timestep_array = sub_epoch_array*N_steps
        
    # Plot result
    plot_output_dir = '{}plots/{}/'.format(base_path, model_name)
    dir_exists = os.path.isdir(plot_output_dir)
    if not dir_exists:
        os.mkdir(plot_output_dir)
    
    NaN_values_x = np.where(np.isnan(loss))[0]
    NaN_values_y = np.zeros_like(NaN_values_x)
        
    fig, ax = plt.subplots()
    plt.plot(timestep_array, loss)
    plt.scatter(NaN_values_x, NaN_values_y, color='r', label='NaN')
    plt.legend()
    plt.title('Training loss')
    plt.ylabel('Loss')
    plt.xlabel('Time steps')
    plt.yscale('log')
    plt.savefig('{}{}_training_loss.pdf'.format(plot_output_dir, 
                model_name), bbox_inches='tight')



# Get cpu or gpu device for training. Note: my MacBook doesn't have CUDA gpu    
device = "cpu"

base_path = '/Users/mwinter/Documents/python_code/NN_measurements/'    
model_name = 'reg_dMNIST_repeat4_w32_d6_b2048'
input_dir = '{}models/{}/'.format(base_path, model_name)

N_inputs = 196
N_outputs = 10
h_layer_widths = [32, 32, 32, 32, 32, 32]
batch_size = 2048
dataset = 'downsampled_MNIST'
loss_function = 'CrossEntropy'

params = {'N_inputs':N_inputs,
          'N_outputs':N_outputs,
          'h_layer_widths':h_layer_widths,
          'batch_size':batch_size,
          'dataset':dataset,
          'loss_function':loss_function}
    
models, epochs = load_models(input_dir, model_name, params, device)
print('Loaded epochs from {}'.format(input_dir))

plot_training_loss(models, epochs, model_name, params, device, base_path)