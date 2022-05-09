#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 19:57:27 2022

@author: mwinter
"""

# A script to make fake loss curves to test Sophie's code

import numpy as np
from matplotlib import pyplot as plt
import torch
plt.close('all')

N = 1000
mu = 700
sigma = 10

x = np.arange(N) 
f = 1.0 - 1.0/(1.0 + np.exp(-(x-mu)/sigma))

fig, ax = plt.subplots()
plt.plot(x, f)
ax.set_ylabel("Loss value (-)")
ax.set_xlabel("Time steps (-)")
fig.suptitle(f"Fake loss curve with cliff at 700 time steps")
fig.savefig("fakelosscurve.jpg")


f_tensor = torch.Tensor(f)

torch.save(f_tensor, 'fake_loss_curve.pt')


