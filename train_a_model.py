#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 11:55:30 2021

@author: mwinter
"""

# Train and save a model

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import time
import random
from NN import NeuralNetwork
import json
import os
import sys
import numpy as np

start = time.time()
plt.close('all')

# Load the params json that contains the widths of each layer
def load_model_params(filepath):
    with open(filepath) as infile:
        params = json.load(infile)

    return params

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
        
        # Check device
        if epoch == epochs[0]:
            print('Current device = ', device)
            for l in range(0, len(model.net), 2):
                l_weight = model.net[l].weight
                print('Layer {} device '.format(l), l_weight.device)
        
        models.append(model)
    
    return models, epochs

# Load a dataset
def load_dataset(dataset, batch_size, base_path):
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
    elif dataset in ['binary_MNIST']:
        sample_img = training_data[0][0]
        N_inputs = sample_img.shape[0]*sample_img.shape[1]
    else:
        sample_img = training_data[0][0][0, :, :]
        N_inputs = sample_img.shape[0]*sample_img.shape[1]
    
    if dataset in ['teacher', 'binary_MNIST']:
        N_outputs = len(training_data[0][1])
    
    else:
        unique_labels = []
        for t in training_data:
            try:
                label = t[1].item()
            except AttributeError:
                label = t[1]
                
            if label not in unique_labels:
                unique_labels.append(label)
                
        N_outputs = len(unique_labels)
    
    # Deal with batch size = dataset size case
    if batch_size==-1:
        batch_size = len(training_data)
    
    train_dataloader = DataLoader(training_data, batch_size=batch_size, 
                                  shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, 
                                 shuffle=True, drop_last=True)
    
    return (train_dataloader, test_dataloader, training_data, test_data, 
            N_inputs, N_outputs)

# Define training process
def train(dataloader, model, loss_fn, optimizer, device):
        
    size = len(dataloader.dataset)
    model.train()
    # model.train() tells the model it is in "train" mode. It does not train the model.
    # This is necessary because some model functions act differently on the
    # training and testing datasets.

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # pytorch has different Tensor types for use on GPUs or CPUs. This casts
        # X (the image) and y (the label) to the right type.

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y) # Access value of loss with loss.item()

        # Backpropagation
        # optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None  # Supposedly this is faster 
        # Sets all gradients to zero (stops gradients accumulating over multiple passes)

        loss.backward() # Calculates dloss/dx for all params x, and adds dloss/dx to x.grad.
        optimizer.step() # Does x += -lr * x.grad

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Define testing process
def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad(): # Turns off gradient calculations. This saves time.
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

# Calculate value of loss function
def evaluate_loss(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    model.eval()
    loss = 0
    with torch.no_grad(): # Turns off gradient calculations. This saves time.
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()

    
    if num_batches>0:
        loss /= num_batches
    else:
        loss = np.inf
        
    return loss

# Save json of model params
def save_model_params(params, outpath):
    with open(outpath, 'w') as outfile:
        json.dump(params, outfile)

def split_path(ic_path):    
    chunks = ic_path.split('/')
    ic_path += '/'

    return ic_path, chunks[-1]

def read_in_std_data_from_teacher_dataset(base_path):
    path = '{}models/teacher/teacher_noise.txt'.format(base_path)
    with open(path, 'r') as infile:
        lines = infile.readlines()
        str_std = lines[0][12:]
        try:
            std = float(str_std)
        except ValueError:
            print('COULD NOT CAST ERROR STR_STD TO FLOAT\n')
    
    path = '{}models/teacher/teacher_prior_std.txt'.format(base_path)
    with open(path, 'r') as infile:
        lines = infile.readlines()
        str_std = lines[0][21:]
        try:
            prior_std = float(str_std)
        except ValueError:
            print('COULD NOT CAST PRIOR STR_STD TO FLOAT\n')
    
    return std, prior_std

# Define particular MSELoss for temperature measurements
def make_special_MSELoss(std):

    def loss(output, target):
        return (1.0/(2.0*std**2))*torch.mean((output - target)**2)
        
    return loss

def make_quadratic_hinge_loss():
    
    def quadratic_hinge(output, target):
        return (nn.HingeEmbeddingLoss()(output, target))**2
    
    return quadratic_hinge

if __name__ == "__main__":
    # Check whether this is running on my laptop
    current_dir = os.getcwd()
    if current_dir[:14] == '/Users/mwinter':
        base_path = '/Users/mwinter/Documents/python_code/NN_measurements/'
    else:
        base_path = '/home/phys/20214003/python_code/NN_measurements/'
    
    # Get cpu or gpu device for training. Note: my MacBook doesn't have CUDA gpu    
    if torch.cuda.is_available():
        device = "cuda"
        print('Training with CUDA GPU.')
    else:
        device = "cpu"
        print('Training with CPU.')
    
    # Load any command line arguments (sys.argv has len 1 if no args provided)
    if len(sys.argv)==7:
        model_stub = sys.argv[6]
        
        if model_stub == 'binary_mnist_rbL':
            rate = sys.argv[1]
            b = sys.argv[2]
            L2 = sys.argv[3]
            
            if sys.argv[4]=='True':
                start_from_existing_model = True
            else:
                start_from_existing_model = False
            
            if sys.argv[5]=='True':
                start_from_param_file = True
            else:
                start_from_param_file = False
                
            model_name = model_stub + '_rate{}_b{}_L2{}'.format(rate, b, L2)
        
        else:
            w = sys.argv[1]
            d = sys.argv[2]
            b = sys.argv[3]
            
            if sys.argv[4]=='True':
                start_from_existing_model = True
            else:
                start_from_existing_model = False
            
            if sys.argv[5]=='True':
                start_from_param_file = True
            else:
                start_from_param_file = False
            
            model_stub = sys.argv[6]
            
            model_name = model_stub + '_w{}_d{}_b{}'.format(w, d, b)
        
        print('Running with user provided arguments')
        print(sys.argv)
        
    elif len(sys.argv) == 5:
        student_N = sys.argv[1]
        
        if sys.argv[2]=='True':
            start_from_existing_model = True
        else:
            start_from_existing_model = False
        
        if sys.argv[3]=='True':
            start_from_param_file = True
        else:
            start_from_param_file = False
        
        model_stub = sys.argv[4]
        
        model_name = model_stub + '_{}'.format(student_N)
        
        print('Running with user provided arguments')
        print(sys.argv)
        
    else:
        w = 256
        d = 6
        b = 45000
        start_from_existing_model = False
        start_from_param_file = True
        # model_name = 'downsampled_MNIST_long_w{}_d{}_b{}'.format(w, d, b)
        model_name = 'binary_mnist_load_test'
        print('Running with default arguments')


    ### Load or create model ###
    start_epoch = -1
    
    model_output_dir = '{}models/{}/'.format(base_path, model_name)
    
    print('start_from_existing_model = {}'.format(start_from_existing_model))
    print('start_from_param_file = {}'.format(start_from_param_file))
    
    if start_from_existing_model:
        print('Loading existing model {}'.format(model_name))
        input_dir = '{}models/{}/'.format(base_path, model_name)
        params = load_model_params('{}{}_model_params.json'.format(input_dir, 
                                                                model_name))
        
        hidden_layers = params['h_layer_widths']
        loss_function = params['loss_function']
        learning_rate = params['learning_rate']
        L2_penalty = params['L2_penalty']
        batch_size = params['batch_size']
        dataset = params['dataset']
        
        (train_dataloader, test_dataloader, training_data, test_data, 
            N_inputs, N_outputs) = load_dataset(dataset, batch_size, base_path)
        
        models, epochs = load_models(input_dir, model_name, params, device)
        model = models[-1]
        start_epoch = epochs[-1]
        
        print('Starting from epoch {}'.format(start_epoch+1))
    
    elif start_from_param_file:
        print('Creating model from parameter file')
        input_dir = '{}models/{}/'.format(base_path, model_name)
        params = load_model_params('{}{}_model_params.json'.format(input_dir, 
                                                                model_name))
        hidden_layers = params['h_layer_widths']
        
        loss_function = params['loss_function']
        learning_rate = params['learning_rate']
        L2_penalty = params['L2_penalty']
        batch_size = params['batch_size']
        dataset = params['dataset']
        
        (train_dataloader, test_dataloader, training_data, test_data, 
            N_inputs, N_outputs) = load_dataset(dataset, batch_size, base_path)
                
        if 'initial_condition_path' in params:
            ic_path = params['initial_condition_path']
            ic_input_dir, ic_model_name = split_path(ic_path)
            print('input_dir = ', ic_input_dir)
            print('model_name = ', ic_model_name)
            models, _= load_models(ic_input_dir, ic_model_name, params, device)
            model = models[0]
            print('Loading from initial condition:')
            print(ic_path)
        else:
            model = NeuralNetwork(n_in=N_inputs, n_out=N_outputs,
                              h_layer_widths=hidden_layers).to(device)
        
        # Save initial state
        dir_exists = os.path.isdir(model_output_dir)
        if not dir_exists:
            os.mkdir(model_output_dir)
            
        torch.save(model.state_dict(),'{}{}_model_epoch_0.pth'.format(
            model_output_dir, model_name))
        print('Saved PyTorch Model State to '+
              '{}{}_model_epoch_0.pth'.format(model_output_dir, model_name))
        
    else:
        print('Creating new model.')
                
        loss_function = 'CrossEntropy'
        learning_rate = 1e-3
        batch_size = b
        width = w
        depth = d
        dataset = 'downsampled_MNIST'
        L2_penalty = 10**(-3)
        
        (train_dataloader, test_dataloader, training_data, test_data, 
            N_inputs, N_outputs) = load_dataset(dataset, batch_size, base_path)
        
        hidden_layers = [width]*depth
        model = NeuralNetwork(n_in=N_inputs, n_out=N_outputs,
                              h_layer_widths=hidden_layers).to(device)
        
        # Save initial state
        model_output_dir = '{}models/{}/'.format(base_path, model_name)
        dir_exists = os.path.isdir(model_output_dir)
        if not dir_exists:
            os.mkdir(model_output_dir)
            
        torch.save(model.state_dict(),'{}{}_model_epoch_0.pth'.format(
            model_output_dir, model_name))
        print('Saved PyTorch Model State to '+
              '{}{}_model_epoch_0.pth'.format(model_output_dir, model_name))
    
    print('h_layer_widths = {}'.format(params['h_layer_widths']))
    print('batch_size = {}'.format(params['batch_size']))
    
    # Check data_output_dir exists
    data_output_dir = '{}measured_data/{}/'.format(base_path, model_name)
    dir_exists = os.path.isdir(data_output_dir)
    if not dir_exists:
        os.mkdir(data_output_dir)
    
    # Check how many parameters the network has
    N_total_params = sum(p.numel() for p in model.parameters() 
                               if p.requires_grad)
    print('Total number of parameters = ', N_total_params)
    print('Size of training set = ', len(training_data))        
    
    # Define loss function
    std, prior_std = read_in_std_data_from_teacher_dataset(base_path)
        
    if loss_function == 'CrossEntropy':
        loss_fn = nn.CrossEntropyLoss()
        w_decay = params['L2_penalty']
    
    elif loss_function == 'MSELoss':
        loss_fn = nn.MSELoss()
        w_decay = params['L2_penalty']
    
    elif loss_function == 'Hinge':
        loss_fn = nn.HingeEmbeddingLoss()
        w_decay = params['L2_penalty']\
            
    elif loss_function == 'quadratic_hinge':
        loss_fn = make_quadratic_hinge_loss()
        w_decay = params['L2_penalty']
    
    elif loss_function == 'special_MSELoss':
        loss_fn = make_special_MSELoss(std)
        P = len(training_data)
        w_decay = 1.0/(2.0*N_total_params*prior_std*prior_std)
    
    else:
        print('PROVIDE A LOSS FUNCTION')
    
    # Define the optimizer. weight_decay>0 adds L2 regularisation to weights
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, 
                                weight_decay=L2_penalty)
    
    # Save parameters
    model_params = {}
    model_params['batch_size'] = batch_size
    model_params['dataset'] = dataset
    model_params['loss_function'] = loss_function
    model_params['learning_rate'] = learning_rate
    model_params['L2_penalty'] = L2_penalty
    model_params['N_inputs'] = N_inputs
    model_params['N_outputs'] = N_outputs
    model_params['h_layer_widths'] = hidden_layers
    
    save_model_params(model_params,'{}{}_model_params.json'.format(
            model_output_dir, model_name))
    
    # Check how many parameters the network has
    N_total_params = sum(p.numel() for p in model.parameters() 
                               if p.requires_grad)
    print('Total number of parameters = ', N_total_params)
    print('Size of training set = ', len(training_data))
    
    ### Train for some number of epochs ###
    epochs = 20000
    # epochs = 500000
    # epochs = 100000
    test_loss = []
    train_loss = []
    epoch_list = []
    for t in range(epochs):
        t += start_epoch + 1
        print(f"Epoch {t+1}\n-------------------------------")

        train(train_dataloader, model, loss_fn, optimizer, device)
                
        # Save every 100th epoch
        if (t+1)%100==0:
            torch.save(model.state_dict(), 
                       '{}{}_model_epoch_{}.pth'.format(model_output_dir, 
                                                               model_name, t))
            train_loss.append(evaluate_loss(train_dataloader, model, loss_fn, 
                                            device))
            
            train_loss_outpath = data_output_dir + 'train_loss.txt'
            out_string = '{} {}\n'.format(t, train_loss[-1])
            try:
                with open(train_loss_outpath, 'a') as f:
                        f.write(out_string)
            except FileNotFoundError:
                with open(train_loss_outpath, 'w') as f:
                        f.write(out_string)
        
            test_loss.append(evaluate_loss(test_dataloader, model, loss_fn, 
                                            device))
            test_loss_outpath = data_output_dir + 'test_loss.txt'
            out_string = '{} {}\n'.format(t, test_loss[-1])
            try:
                with open(test_loss_outpath, 'a') as f:
                        f.write(out_string)
            except FileNotFoundError:
                with open(test_loss_outpath, 'w') as f:
                        f.write(out_string)
                        
            epoch_list.append(t)
    
    print("Done!")
    
    # Rescale time to be in timesteps not epochs
    epoch_array = np.array(epoch_list)
    N_steps = len(train_dataloader)
    timestep_array = epoch_array*N_steps
    
    # Plot losses
    plot_output_dir = '{}plots/{}/'.format(base_path, model_name)
    dir_exists = os.path.isdir(plot_output_dir)
    if not dir_exists:
        os.mkdir(plot_output_dir)
    
    if start_epoch == -1:
        start_epoch = 0 # for plotting
    fig, ax = plt.subplots()
    plt.plot(timestep_array, test_loss)
    plt.title('Loss on test set')
    plt.xlabel('Time/steps')
    plt.savefig('{}{}_testing_loss.pdf'.format(plot_output_dir, model_name), 
                bbox_inches='tight')
    
    fig, ax = plt.subplots()
    plt.plot(timestep_array, train_loss)
    plt.title('Loss on training set')
    plt.xlabel('Time/steps')
    plt.savefig('{}{}_training_loss.pdf'.format(plot_output_dir, model_name), 
                bbox_inches='tight')
    
    print('Running time = ', time.time()-start)







