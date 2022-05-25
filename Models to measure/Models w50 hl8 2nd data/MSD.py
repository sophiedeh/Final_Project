# -*- coding: utf-8 -*-
"""
Created on Fri May 13 10:27:15 2022

@author: 20182672
"""
import torch
from Models_w50to100_hl8to10 import NeuralNetwork
import matplotlib.pyplot as plt

training_times = 12
width = 50
hidlay = 8
epochs = 50000
version = 4
dataset_size = 1000
epoch = 0
batch_size = 25 
MSD = []
batch_sizes = []

for u in range(version):
    batch_size = 25
    batch_sizes = []
    average_MSD_per_bs = []
    for i in range(training_times):
        average_MSD_per_epoch = []
        epoch = 0
        initial_weights = []
        MSD_per_epoch = []
        epochs = []
        for t in range(500-1): 
            model = NeuralNetwork(width=width,hidlay=hidlay)
            model.load_state_dict(torch.load(f"model_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epoch}_tt{training_times}.pth")) 
            print(model.parameters())
    
            weights = []
            for name, param in model.named_parameters():
                if param.requires_grad: # is for freezing, doesn't seem necessary now?
                    for i in range(hidlay*2+1):
                        if name == f"linear_relu_stack.{i}.weight":
                            if epoch == 0:
                                initial_weights.append(param.data)
                            else:
                                weights.append(param.data)
            
            Diff_squared = [] #this will create every time step new diff squared
            if epoch == 0:
                epoch += 100
            else:
                for i in range(len(initial_weights)):
                    sub = torch.sub(initial_weights[i],weights[i]) 
                    squared = torch.square(sub)
                    #summed = torch.sum(squared)
                    #MSDi = torch.mean(squared) #mean MSD of one layer
                    Diff_squared.append(torch.reshape(squared,(-1,))) #array containing diff squared tensor of all layers
                    #import pdb; pdb.set_trace() #to check if shape is correct
                
                Diff_squared_one = torch.cat(Diff_squared)    
                MSD_all_layers = torch.mean(Diff_squared_one)
                MSD_per_epoch.append(MSD_all_layers)
                #MSD_all_layers = torch.DoubleTensor(MSD_all_layers) #MSD of all layers at one particular time
                
            #aver_MSD = torch.mean(MSD) #average MSD of all the layers at that time
            #average_MSD_per_epoch.append(aver_MSD)        
                epochs.append(epoch) #consider that only after epoch 0 MSD is measured
            epoch += 100    
        
        number_of_steps = []
        for i in range(len(epochs)):
            number_of_steps.append((i+1)*(dataset_size/batch_size))
        
        fig = plt.figure()
        ax = fig.subplots()
        fig.suptitle(f"MSD of bs{batch_size} w{width} hl{hidlay} v{u+1}")       
        ax.plot(number_of_steps, MSD_per_epoch,'-', marker='o')
        ax.set_xlabel("Number of steps (-)")
        ax.set_ylabel("MSD of the layers over time (-)")
        plt.savefig(f"MSD of bs{batch_size} w{width} hl{hidlay} v{u+1}.png",bbox_inches='tight')
        
        
    #     average_MSD_per_epoch = torch.DoubleTensor(average_MSD_per_epoch)
    #     average_MSD_per_bs.append(torch.mean(average_MSD_per_epoch))
    #     torch.save(average_MSD_per_bs,f"MSD_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}.pth")
    #     batch_sizes.append(batch_size)
    #     batch_size += 25
    # fig = plt.figure()
    # ax = fig.subplots()
    # fig.suptitle(f"MSD per batch size w{width} hl{hidlay} v{u+1}")
    # ax.plot(batch_sizes, average_MSD_per_bs,'-', marker='o')
    # ax.set_xlabel("Batch size (-)")
    # ax.set_ylabel("Average MSD (-)")
    # #plt.scatter(batch_sizes, noise_per_bs)
    # plt.show()
    # plt.savefig(f"Average MSD per batch size w{width} hl{hidlay} v{u+1}.png",bbox_inches='tight')
  
    
        
    
  # for name, param in model.named_parameters():
  #     if param.requires_grad: #check
  #         print(name, param.data)
  #         for i in range((hidlay+1)*2):
  #         if name == linear_relu_stack.i.weight:
  #             if epoch == 0:
  #                 initial_weights.append(param.data)
  #             else    
  #                 weights.append(param.data)
  
  
  # for i in range(0 10 2) or loop all layers: try, except
  #import pdb; pdb.set_trace()
  
  
  # if epoch == 0:
  #     initial_weights.append()
  # else:
  #     weights.append()
  
  # initial_weights = torch.Tensor(initial_weights)
  # weights = torch.Tensor(weights)    
    
