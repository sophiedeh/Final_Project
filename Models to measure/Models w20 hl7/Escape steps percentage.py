# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:15:11 2022

@author: 20182672
"""
import torch
import matplotlib.pyplot as plt

dataset_size = 1000
epochs = 50000
training_times = 12

width = 20
hidlay = 7
version = 4

for u in range(version):
    batch_size = 25
    plt.figure()
    plt.title(f"Loss values of network with width {width}, hidden layers {hidlay} and version {u+1}")
    escape_steps_values = []
    escape_positions = []
    batch_sizes = []
    for i in range(training_times):
        number_of_steps = []
        
        # Load loss values
        loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")
        loss_values = []
        
        for i in range(0, 50000, 100):
            loss_values.append(loss_values_train[i])
            number_of_steps.append((i+1)*(dataset_size/batch_size))
        # for i in range(len(loss_values_train)):
        #     number_of_steps.append((i+1)*(dataset_size/batch_size))
        
        #fig = plt.figure()
        #ax = fig.subplots()
        	
        escapes_for_value = []
        for x in range(len(loss_values_train)):
            escapes_for_value.append((x+1)*(dataset_size/batch_size))
                
        t = 1
        while loss_values_train[t] >= 0.15*loss_values_train[0]:
            t += 1
            escape_step_value = loss_values_train[t+1]
            escape_pos = escapes_for_value[t+1]
        escape_steps_values.append(escape_step_value)
        escape_positions.append(escape_pos)
        
        plt.plot(number_of_steps, loss_values,'-',escape_pos)
        plt.yscale('log')
        #plt.legend(f"{batch_size}")
        plt.xlabel("Number of steps (-)")
        plt.ylabel("Loss value (-)")
        plt.scatter(escape_pos, escape_step_value)
        plt.xlim([60000,120000])
        plt.ylim(bottom=((10)**(-2)),top=10**0)
        
        batch_sizes.append(batch_size)
        batch_size += 25   
    plt.legend(['Bs 25','Bs 50', 'Bs 75','Bs 100', 'Bs 125', 'Bs 150', 'Bs 175', 'Bs 200','Bs 225', 'Bs 250', 'Bs 275', 'Bs 300'])
    plt.show()
    plt.savefig(f"All loss values of w{width} hl{hidlay} v{u+1}.pdf",bbox_inches='tight')
    
    plt.plot(batch_sizes, escape_positions,'-', marker='o')
    plt.title(f"Escape steps percentage of w{width} hl{hidlay} v{u+1}")
    plt.show()
    plt.savefig(f"Escape steps percentage of w{width} hl{hidlay} v{u+1}.pdf",bbox_inches='tight')







# for u in range(version):
#     batch_size = 25
#     plt.figure()
#     plt.title(f"Loss values of network with escape point width {width}, hidden layers {hidlay} and version {u+1}")
#     for i in range(training_times):
#         number_of_steps = []

#         # Load loss values
#         loss_values_train = torch.load(f"Loss_values_train_bs{batch_size}_w{width}_hl{hidlay}_v{u+1}_ds{dataset_size}_e{epochs}_tt{training_times}.pt")

#         # for i in range(len(loss_values_train)):
#         #     number_of_steps.append((i+1)*(dataset_size/batch_size)) # i + 1 since first value corresponds to first epoch so not zero epoch
        
#         loss_values = []
#         for i in range(0, 50000, 100):
#             loss_values.append(loss_values_train[i])
#             number_of_steps.append((i+1)*(dataset_size/batch_size))
        
#         escapes_for_value = []
#         for x in range(len(loss_values_train)):
#             escapes_for_value.append((x+1)*(dataset_size/batch_size))
        
#         t = 1
#         while loss_values_train[t] >= 0.9*loss_values_train[0]:
#             t += 1
#         else:
#             escape_step_value = loss_values_train[t]
#             escape_pos = escapes_for_value[t]
#             break
            
#         # loss_values = []
#         # for i in range(0, 50000, 100):
#         #     loss_values.append(loss_values_train[i])
#         #     number_of_steps.append((i+1)*(dataset_size/batch_size))
        
#         plt.plot(number_of_steps, loss_values,'-')# escape_pos, escape_step_value, marker = 'o')
#         plt.yscale('log')
#         #plt.legend(f"{batch_size}")
#         plt.xlabel("Number of steps (-)")
#         plt.ylabel("Loss value (-)")
#         #plt.scatter(escape_pos, escape_step_value)
#         #plt.xlim([85000,140000])
#         #plt.ylim(10**(-3),0)
        
#         batch_size += 25
#     plt.legend(['Bs 25','Bs 50', 'Bs 75','Bs 100', 'Bs 125', 'Bs 150', 'Bs 175', 'Bs 200','Bs 225', 'Bs 250', 'Bs 275', 'Bs 300'])
#     plt.show()
#     plt.savefig(f"Loss values with escape steps of w{width} hl{hidlay} v{u+1}.pdf",bbox_inches='tight')
                
                
                