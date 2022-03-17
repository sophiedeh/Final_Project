# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 09:20:47 2022

@author: 20182672
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from Create_5_models_w50d8 import NeuralNetwork

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

     # Compute prediction error
        pred = model(X) #prediction error vector
        loss = loss_fn(pred, y) #I believe summing over all elements is implementedn in loss function

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    loss_values_train.append(running_loss/len(dataloader))
                      
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax()).type(torch.float).sum().item()
    test_loss /= num_batches
    loss_values_test.append(test_loss)
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# def quadratic_hinge_loss_function(predicted,correct):
#     # specifying the batch size
#     size_dataset = predicted.size()[0] 
#     # calculating the log of softmax values 
#     delta, small_loss = 0
#     for i in size_dataset:
#         if predicted.argmax(1) == correct.argmax():
#             delta = 1 - predicted[i,:]*correct[i,:]
    
        
              
#     my_outputs = F.log_softmax(my_outputs, dim=1)  
#     selecting the values that correspond to labels
#     my_outputs = my_outputs[range(my_batch_size), my_labels] 
#     returning the results
#     return -torch.sum(my_outputs)/number_examples    

training_times = 5 #amount of how many times to train data
width = 50 #amount of nodes
depth = 8 #amount of layers

# Load training data from own script. 
training_data = torch.load('mini_pca_train.pt') #use bigger data set - check if this happened in variable explorer

# Load test data from own script.
test_data = torch.load('mini_pca_test.pt')

# Define size 
dataset_size = len(training_data)

# Define the batch size
batch_size = 200

for u in range(training_times):       
    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True,drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    model = NeuralNetwork(width=width)
    model.load_state_dict(torch.load(f"initial_model_{u+1}_w{width}_d{depth}.pth"))    
    
    loss_fn = nn.CrossEntropyLoss() #square error or hinge loss
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    loss_values_train = []
    loss_values_test = []
  
    start = time.time()
    epochs = 100000
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
        end = time.time()
        elapsed_time = end - start
    print("Done!")
    print(f"Elapsed time {elapsed_time}\n-------------------------------")
        
    number_of_steps = []
    for i in range(len(loss_values_train)):
        number_of_steps.append(i*(dataset_size/batch_size))
        
    fig, axs = plt.subplots(2,sharex=True)
    fig.suptitle(f"Train vs Test Losses Version: {u+1}")
    axs[0].plot(number_of_steps, loss_values_train,'-')
    axs[0].set_title('Train')
    axs[1].plot(number_of_steps, loss_values_test,'-')
    axs[1].set_title('Test')

    for ax in axs.flat:
        ax.set(xlabel='Number of steps', ylabel='Loss')
        ax.set_yscale('log')

    for ax in axs.flat:
        ax.label_outer()

    plt.show()
    fig.savefig(f"Plot_bs{batch_size}_w{width}_d{depth}_version{u+1}_e{epochs}_tt{training_times}.pdf")
    
    torch.save(model.state_dict(), f"model_bs{batch_size}_w{width}_d{depth}_version{u+1}_e{epochs}_tt{training_times}.pth")
    print(f"Saved PyTorch Model State to model_bs{batch_size}_w{width}_d{depth}_version{u+1}_e{epochs}_tt{training_times}.pth")
    
    #batch_size += 50