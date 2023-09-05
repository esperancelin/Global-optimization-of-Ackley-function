# -*- coding: utf-8 -*-
"""
samples class contains all the method to handle the samples generated from optimizatio process
@author: rongh
"""

import numpy as np
from sklearn.cluster import KMeans
from warnings import catch_warnings
from warnings import simplefilter
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from BayesOpt  import *
from meta import SamplesMeta

class samples:
    def __init__(self,n_dim, data=[], fom=[], boundary=[]):
        self.data_list = data  #################data and fom are stored as list
        self.fom_list=fom
        self.boundary=boundary
        self.root_data=[]
        self.root_fom=[]
        self.n_dim=n_dim

          
    def n(self):
        return len(self.data_list)
    def data(self):   
        return np.array(self.data_list)######data and fom are stored as array (num, dim)
    def fom(self):
        return np.array(self.fom_list) 
    def best_data_fom(self):
        best_fom=max(self.fom_list)
        idx=self.fom_list.index(best_fom)
        best_data=self.data_list[idx]
        return best_data, best_fom
        
    def fmu(self):
        return np.sum(self.fom())/self.n()
    
    def splitable(self):
        if self.n()>SamplesMeta.split_thres:
            return True  
        else :
            return False
     
    def split_sample(self,n_cluster=2, random_state=1):      
        kmeans = KMeans(n_clusters=n_cluster, random_state=random_state, n_init="auto",verbose=0).fit(self.data)
        cluster=kmeans.labels_
        ps=[]
        pf=[]
        qs=[]
        qf=[]
        for s, f, label in zip(self.data_list, self.fom_list, cluster):
            if label==0:
                ps.append(s)
                pf.append(f)
            else:
                qs.append(s)
                qf.append(f)      
        sample1=samples(ps, pf)
        sample2=samples(qs, qf)
        
        if sample1.fmu()>sample2.fmu():#######################make sure the bigger fmu is sample 1
        
            return sample1,sample2
        
        else:
            return sample2, sample1
        
    def split_replace(self,n_cluster=2, random_state=1):
        kmeans = KMeans(n_clusters=n_cluster, random_state=random_state, n_init="auto",verbose=0).fit(self.data())
        cluster=kmeans.labels_       
        ps=[]
        pf=[]
        qs=[]
        qf=[]
        for s, f, label in zip(self.data_list, self.fom_list, cluster):
            if label==0:
                ps.append(s)
                pf.append(f)
            else:
                qs.append(s)
                qf.append(f)
        qf_av=np.max(np.array(qf))
        pf_av=np.max(np.array(pf))

        
        if qf_av>pf_av:############keep the list with the highest fom value inside     
            self.data_list=qs
            self.fom_list=qf
        else:
            self.data_list=ps
            self.fom_list=pf

    def add_sample(self, data, fom):
        self.data_list.append(data)
        self.fom_list.append(fom)

    
    def init_random_samples(self, n_dim, num, func, random_seed=None):
        if random_seed is not None:
            np.random.RandomState(seed=random_seed)
        x=(np.random.rand(n_dim,num)-0.5)*20
       
        b=func(x)
        
        self.data_list=x.T.tolist()
        self.fom_list=b.tolist()
  
    def get_outcome(self):##return the final score of the search, 
        return self.fmu()



