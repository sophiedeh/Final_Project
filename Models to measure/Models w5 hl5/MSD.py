# -*- coding: utf-8 -*-
"""
Created on Fri May 13 10:27:15 2022

@author: 20182672
"""
import torch
from Models_w50to100_hl8to10 import NeuralNetwork

training_times = 12
width = 5
hidlay = 5
epochs = 50000
version = 4
dataset_size = 1000
epoch = 0
batch_size = 25 
weights = []
initial_weights = []
MSD = []

for i in range(training_times): 
    for u in range(version):
        for t in range(500): 
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            # print(f"Using {device} device")
            
            # model = NeuralNetwork(width=width,hidlay=hidlay).to(device)
            # model.load_state_dict(torch.load(f"model_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{t}_tt{training_times}.pth", map_location=torch.device(device)))  
            
            model = NeuralNetwork(width=width,hidlay=hidlay)
            model.load_state_dict(torch.load(f"model_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epoch}_tt{training_times}.pth")) 
            print(model.parameters())
    
            for name, param in model.named_parameters():
                if param.requires_grad: # is for freezing, doesn't seem necessary now?
                    for i in range(hidlay*2+1):
                        if name == f"linear_relu_stack.{i}.weight":
                            if epoch == 0:
                                initial_weights.append(param.data)
                            else:
                                weights.append(param.data)
                            
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
            if epoch == 0:
                epoch += 100
                break
            else:
                for i in range(len(initial_weights)):
                    sub = torch.sub(initial_weights[i],weights[i]) 
                    squared = torch.square(sub)
                    summed = torch.sum(squared)
                    MSDi = (1/len(initial_weights)) * summed
            
                    MSD.append(MSDi)
                    epoch += 100
        torch.save(MSD,f"MSD_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}.pth")
        epoch = 0
    batch_size += 25
    
    
