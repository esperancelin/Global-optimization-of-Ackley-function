# -*- coding: utf-8 -*-
"""

Main optimization loop
@author: rongh
"""
import numpy as np
import matplotlib.pyplot as plt
from Optimization import *
from meta import ackley
import pandas as pd

n_dim=13
f=ackley(n_dim)


def write_fom_his(fom_his):
    with open('grad_fom_his',"a") as f:
        f.write(str(fom_his)+'\n')


##########################################
his=[]
opt=Optimization(f, n_dim)
opt.run(150)

print(i)
his.append(opt.his)
write_fom_his(opt.his)


########################################


