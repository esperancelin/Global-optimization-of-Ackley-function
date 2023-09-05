# -*- coding: utf-8 -*-
"""
Optimization classs, main control module
@author: rongh
"""
import numpy as np
from samples import samples
from BayesOpt  import BayesOpt
from mcts import MCTS
from copy import deepcopy
import time

class Optimization():
    def __init__(self,func, dims=10, ):
        self.n_dim        = dims 
        self.func          =func
        self.n_init_spl    =50
        self.his       =[]
        self.best      =[]
        self.fom=[]


    def run(self, itrs):
        n_cluster=2
        self.his=[]
        x_sample_original=samples(self.n_dim )
        x_sample_original.init_random_samples(n_dim=self.n_dim , num=self.n_init_spl, func=self.func)
        _, self.fom=x_sample_original.best_data_fom()
        self.his.append(self.fom)
        
        
        
        for itr in range(itrs):
            start_time = time.time()
            x_sample=deepcopy(x_sample_original)
            state_his=[]
            

            ###########################################################MCTS method
            mcts = MCTS(x_sample,x_sample_original.data_list.copy(), x_sample_original.fom_list.copy())
            while x_sample.splitable(): ########keep split the sample until we find a good split  
                mcts.search()
                next_split = mcts.suggest_best_move()
                state_his.append(next_split)
                mcts.update(next_split) ##inform the mcts after splitting
                x_sample.split_replace(n_cluster=2, random_state=next_split)
            ############################################################################################
        
            picked_sample=deepcopy(x_sample)
            
            mct_time=time.time() - start_time
        
            st=2 if itr>30 else 0
            # st=2
            BO=BayesOpt(x_sample_original.data_list.copy(), x_sample_original.fom_list.copy(), picked_sample.data_list.copy(), picked_sample.fom_list.copy(), self.n_dim, sample_tech=st)
            BO.fit()
            
                
            new_sample=BO.suggest() #########sampleing happen at sugggest
            est_fom,_=BO.surrogate([new_sample])
            new_sample_fom=self.func(new_sample)
            _,oldfom=x_sample_original.best_data_fom()
        
            #print("####################iteration %s ,MCT time %s " %(itr,(time.time() - start_time)))
            print("")
            print("########################## iteration: {}, MCTS time: {:.1f} seconds".format(itr,mct_time))
         
            print(" old        expct      real: ")
            print("%.3f    " % oldfom, "%.3f    " % est_fom[0], "%.3f    " % new_sample_fom)
            
            x_sample_original.add_sample(new_sample, new_sample_fom)
            self.best, self.fom=x_sample_original.best_data_fom()
            self.his.append(self.fom)
        print("")
  
        print("########### Optimization Done ###########")
        print("The best FOM is: {:.3f}".format(self.fom) )
        print("The best x is {}".format(self.best))