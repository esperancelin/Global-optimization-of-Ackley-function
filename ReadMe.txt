This is the supplementary information 3 to "Schordinger's red beyond 65,000 pixel per inch by multipolar interaction in freeform meta-atom through efficient neural optimizer". We implemented and improved LaMCTS (Wang, L., Fonseca, R., & Tian, Y. (2020) Advances in Neural Information Processing Systems, 33, 19511-19522) to do Monte Carlo Tree Search to partition the search area to achieve higher performance optimization.

The target function is a 13D achkey function with minimum 0 at [1,1,1,1,1...]

The following dependencies must be met

NUMPY (>1.24.0)
SCIPY (>1.16.0)
scikit-learn (>1.2.0)

The code contains the following modules:

main.py           @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ This is the main function that defines the object function and call the optimization algorithm
Optimization.py   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ This is the main control module that controls the iterations of the optimization
mcts.py           @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ This is the module that handles the Monte Carlo Tree Search
samples.py        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ This is the module that handles the samples, including geting the FOM, store the total sample set, sample splitting, it is called by MCTS 
BayesOpt.py       @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ This module runs the local Bayesian optimization to provide the next sample to probe
meta.py           @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ This file contains the meta parameters of the whole algorithm. Such as the target function,  architecutre of the MCTS, spliting threshold of the samples 

IF you use this code, please consider citing our manuscript. 