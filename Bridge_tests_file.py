# This file is made to test the functions defined in the file bridge.py
#%%

import math
import PF_functions_def as pff
# We import the module PF_functions_def such that it has 
# several important function from the project 
#https://github.com/maabs/Multilevel-for-Diffusions-
#Observed-via-Marked-Point-Processes **
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
import copy
from scipy.sparse import identity
from scipy.sparse import rand
from scipy.sparse import diags
from scipy.sparse import triu

#from sklearn.linear_model import LinearRegression
from scipy.stats import ortho_group
import time
from scipy.stats import norm
import scipy.stats as ss

#%%



x_pf=np.reshape(np.loadtxt("Observations&data/CCPF_lan20_v.txt",dtype=float),(2,len(eles),samples,int(T/d),dim))
