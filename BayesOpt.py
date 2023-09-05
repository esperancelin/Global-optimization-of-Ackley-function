# -*- coding: utf-8 -*-
"""
Baysian optimization module
@author: rongh
"""
from warnings import catch_warnings
from warnings import simplefilter
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from meta import BoMeta
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import random


class BayesOpt:
    def __init__(self,init_sample_x,init_fom,pickedx, pickedfom, n_dim, sample_tech):
        self.X = init_sample_x  ###arrary
        self.y=init_fom
        self.gp=GaussianProcessRegressor(kernel=Matern(nu=BoMeta.Matern_nu), alpha=1e-6, normalize_y=True, n_restarts_optimizer=BoMeta.gprestart)
        self.n_dim=n_dim
        self.sample_tech=sample_tech #############sample technie should be 0,1,2 
        self.pickedx=pickedx
        self.pickedfom=pickedfom

    def get_best(self):
        best_fom=np.max(np.array(self.y))
        best_idx=np.argmax(np.array(self.y))
        best_x=self.X[best_idx]
        return best_x, best_fom
    
    def register(self, sample_x, sample_fom):
        self.X.append(sample_x)
        self.y.append(sample_fom)
    
    def fit(self):
        with catch_warnings():
            simplefilter("ignore")
            self.gp.fit(np.array(self.X), np.array(self.y))
                
    def surrogate(self, X):
         # catch any warning generated when making a prediction
         with catch_warnings():
         # ignore generated warnings
             simplefilter("ignore")
         return self.gp.predict(X, return_std=True)

    def suggest(self):###########################requires a clf object that determines the boundary
        # optimize the acquisition function
    	# random search, generate random samples
        
        randomsamples=self.sampling(num_sample=BoMeta.num_rsample)
        
        if BoMeta.acq=="PI":
            scores = self.acquisition_pi(randomsamples)
        else:
            scores = self.acquisition_ei(randomsamples)
        
        
    	# locate the index of the largest scores
        ix = np.argmax(scores)
  
        return randomsamples[ix,:] ####this is the suggested next sample
    
    def get_para_sampling(self):
        # a=np.array(self.pickedx)
        # b=np.max(a,axis=0)
        c=np.array(self.pickedx)
        best_x, _=self.get_best()
        cov=np.cov(c.T,bias=True)
        return best_x, cov##
    
    def sampling(self, num_sample):
        
        if self.sample_tech==0:
            samples = np.random.rand(num_sample,self.n_dim)
        
        if self.sample_tech==1:
            mu,cov=self.get_para_sampling()
            samples=np.random.multivariate_normal(mu, cov, size=num_sample)        
        
        if self.sample_tech==2:
            mu,cov=self.get_para_sampling()
            sample_itr=0
            
            label=np.zeros(len(self.X))
            picked_x_arrays = np.array(self.pickedx)
            for i, x in enumerate(self.X):
                is_in_list = np.any(np.all(np.array(x) == picked_x_arrays, axis=1))
                if is_in_list:
                    label[i]=1
            clf = make_pipeline(StandardScaler(), SVC(gamma=2, C=1))
                  
            while True:
                sample_itr+=1
                init_samples=np.random.multivariate_normal(mu, cov, size=num_sample*5) 
                
            ############################################initialilzed a SVM to check to sample

                clf.fit(np.array(self.X), label)
            ###############################################start sampling
            
                test=clf.predict(init_samples).reshape(-1,1)
                good_samples=test*init_samples
                

            #################reject the samples outside the SVM prediction    
                idx = np.argwhere(np.all(good_samples[..., :] == 0, axis=1))
                samples = np.delete(good_samples, idx, axis=0)

                
                #print("clean sample size",samples.shape)
                if samples.shape[0]>1000:
                    break
                
                if sample_itr>10:
                    samples=np.random.multivariate_normal(mu, np.random.rand()*np.ones((self.n_dim, self.n_dim)), size=num_sample)
                    break
            
        return np.array(samples)
        
    
    # probability of improvement acquisition function
    def acquisition_pi(self, randomsamples):
     	# calculate the best surrogate score found so far
        yhat, _ = self.surrogate(self.X)
        best = max(yhat)
         	# calculate mean and stdev via surrogate function
        mu, std = self.surrogate(randomsamples)
         	# calculate the probability of improvement
        probs = norm.cdf((mu - best) / (std+1E-9))
        return probs
    
    
    def acquisition_ei(self, x_to_predict,greater_is_better=False, n_params=1):
        """ expected_improvement acuisition
    
        Arguments:
        ----------
            x: array-like, shape = [n_samples, n_hyperparams]
                The point for which the expected improvement needs to be computed.
            gaussian_process: GaussianProcessRegressor object.
                Gaussian process trained on previously evaluated hyperparameters.
            evaluated_loss: Numpy array.
                Numpy array that contains the values off the loss function for the previously
                evaluated hyperparameters.
            greater_is_better: Boolean.
                Boolean flag that indicates whether the loss function is to be maximised or minimised.
            n_params: int.
                Dimension of the hyperparameter space.
    
        """
        _, evaluated_loss=self.get_best()
        
        mu, sigma = self.gp.predict(x_to_predict, return_std=True)
        
        if greater_is_better:
            loss_optimum = np.max(evaluated_loss)
        else:
            loss_optimum = np.min(evaluated_loss)
        
        scaling_factor = (-1) ** (not greater_is_better)
        
        # In case sigma equals zero
        with np.errstate(divide='ignore'):
            Z = scaling_factor * (mu - loss_optimum) / sigma
            expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
            expected_improvement[sigma == 0.0] == 0.0
        
        return -1 * expected_improvement   

    def plot1d(self):
         plt.figure()
         # scatter plot of inputs and real objective function
         plt.scatter(self.X, self.y)
         # line plot of surrogate function across domain
         Xsamples = np.array(np.arange(-10, 10, 0.01))
         #Xsamples = Xsamples.reshape(len(Xsamples), 1)
         ysamples, std = self.surrogate(Xsamples.reshape(-1,1))
         plt.plot(Xsamples, ysamples)
         plt.fill_between(Xsamples, ysamples-1.96*std, ysamples+1.96*std, alpha=0.3 )  
         plt.show()
