# -*- coding: utf-8 -*-

import math
import numpy as np

"""
This file contains the metadata for the whole algorith
MCTSMeta is for MCTS
SamplesMeta is for samples class
BoMeta is for baysian optimization
ackley is the target function. 

"""

class MCTSMeta:
    EXPLORATION = math.sqrt(2)  ##exploration value for MCTS
    n_children=2               ##number of child for each node
    n_itr=10                    

class SamplesMeta:
    split_thres=40 ####the sample is splitable if number is greater than this value
    
class BoMeta:
    Matern_nu=2.5   ##nu value 
    gprestart=5## restart number of gaussian process
    num_rsample=10000  ##number of samples to evaluate for random sampling
    acq="EI"  ##can be EI(expected improvements) or PI(probability of improvements), default is EI
    
class ackley():
    def __init__(self, dims=10):
        self.dims        = dims
        self.lb          = -10 
        self.ub          =  10 
        self.fcall       =0             ##number of cuntion calls

    def __call__(self, individual): 

        self.fcall+=1
        d = len(individual)
        f=20 - 20 * np.exp(-0.2*np.sqrt(1.0/d * sum((x-1)**2 for x in individual))) \
                + np.exp(1) - np.exp(1.0/d * sum(np.cos(2*np.pi*(x-1)) for x in individual))
        return -f
       

