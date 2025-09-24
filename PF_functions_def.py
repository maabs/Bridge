#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:49:24 2022

@author: alvarem

General description: This file stores the functions necessary to
ensemble the PF. Since the taks assigned to the functions are simple
(we could say that the functions are atomized) the PF are highly costumizable. 
For example, we can easily obtain a version for the PF, the PF with Cox observations
with minor tweaks.
"""
#%%
import math
import numpy as np
import matplotlib.pyplot as plt
#import progressbar
from scipy import linalg as la
from scipy.sparse import identity
from scipy.sparse import rand
from scipy.sparse import diags
from scipy.sparse import triu
import copy
#from sklearn.linear_model import LinearRegression
from scipy.stats import ortho_group
import time
from scipy.stats import multivariate_normal
#%%
"""
IMPORTANT: whenever we use this functions for different systems we need to keep in mind that some changes are 
necessary to the function M, M_smooth and M_smooth_W and Cox_PE. 


In this space I summarize the functions, what they do, how do we use them and 
point out details that are important to keep in mind when using them.

g_normal(x,g_pars): This function generates samples from a normal distribution, it's 
used as likelihood distribution

gen_obs(x,l,T,Lambda,Lamb_par,g,g_par): This function generates observations from a
poisson process with intensity function Lambda(x), it's used in the PF with Cox observations

gen_gen_data(T,x0,l,collection_input): This function generates samples from a diffusion
process, it's called general because the collection input is  [dim, b,A,Sig,fi], where b and 
Sig are functions that take on two arguments, it's used in the PF and the PF with Cox observations.

M(x0,b,A,Sig,fi,l,d,N,dim): This function generates samples from a diffusion process,  ITS IMPORTANT
TO KNOW THAT WE NEED TO COMMENT AND UNCOMMENT SOME LINES DEPENDING ON THE PROCESS WE ARE SIMULATING,
THIS IS MADE TO OPTIMIZE THE COMPUTATION FOR SOME DIFFUSIONS.

M_smooth(x0,b,A,Sig,fi,Grad_b,b_numb_par,l,d,N,dim): This function generates samples from a diffusion process 
and also computes the gradient of the drift function. We have new arguments here as Grad_b,b_numb_par. It's 
necessary to compute the gradient inside the function because for the ito integral we need the dW which is lost 
at each step of the euler maruyama discretization. ITS IMPORTANT TO KNOW THAT WE NEED TO COMMENT AND UNCOMMENT
SOME LINES DEPENDING ON THE PROCESS WE ARE SIMULATING, THIS IS MADE TO OPTIMIZE THE COMPUTATION FOR SOME DIFFUSIONS 
AS IN THE FUNCTION M.

M_smooth_W(x0,x_nr,b,A,Sig,fi,Grad_b,b_numb_par,l,d,N,dim): This function is basically the same as the previous one, 
the changes are that we include the argument x_nr which...

b_gbm(x,mu): This function computes the drift of the geometric brownian motion, it's used in the PF and the PF with Cox observations.
It can be used in several contexts since allows several different ranks for x, so it can be used to generate
simulations of the true process and also to compute the drift of the particles, the function works this way as 
long as the the last dimension of the array x corresponds to the dimension of the problem, i.e., d_x, or dx, 
whatever we call it.

Grad_b_gbm(x,mu): computes the gradient of the drift of the geometric brownian motion.
IT'S ONLY VALID FOR THE ONE DIMENSIONAL CASE. The generalization of this function is not made since it would take
and innecessarely large memory, thus we leave this case for a more specialized/personalized code for the GBM.

Sig_gbm(x,fi): This function computes the diffusion of the geometric brownian motion. We use as fi=[Sig,sigs]
which follows a model a Sig@(sigs*x*dW). This function can be used either for simulation of the "true" process or for 
the PFs

Sig_nldt(x,fi): This function computes the diffusion of the nldt process. We use figs as a constant, we can use this function 
for the PFs BUT only when we are in one dimensional settings. 

b_lan(x,df): this function computes the drift of the langevin diffusion, it's used in the PF and the PF with Cox observations. 
THIS FUNCTION IS ONLY VALID FOR ONE DIMENSIONAL PROBLEMS.

b_ou(x,A): This function is basically a generalized matrix vector multiplication, it's used in several dimension with 
several ranks

Grad_b_ou(x,A): computes the gradient of the drift of the Ornstein-Uhlenbeck diffusion. THIS FUNCTION IS ONLY VALID FOR
ONE DIMENSIONAL PROBLEMS.

Grad_b_lan(x,df): This function computes the gradient of the drift of the langevin diffusion. 
THIS FUNCTION IS ONLY VALID FOR ONE DIMENSIONAL PROBLEMS.

Sig_ou(x,fi): This function computes the diffusion of the Ornstein-Uhlenbeck diffusion.
We use as fi=Sig. This function can be used either for simulation of the "true" process or for

g_den(y, x,   g_par  ): This function helps computing the likelihhod density. It can be used for the smoothing too, i.e.,
the rank of x is 4.

log_trans(xA,xB,b,A,Sig,fi,l): This function computes the density of the transition kernel of the diffusion for the smoothing algorithm.

Grad_Log_g_den(y,x,g_par): THis function computes the gradient of the log likelihood density. It is used only for the smoothing algorithm.

Lambda(x,Lamb_par,ax=0): This function computes the intensity function of the poisson process. In Un_cox_PF_functions_def.py the function 
Norm_Lambda is the same as this one. It can be used in multiple dimension thanks to ax=... .

Grad_Lambda(x,Lamb_par,Lamb_par_numb,ax1=0): This function computes the gradient of the intensity function of the poisson process(when the intensity)
is the function Lambda(). Notice that we include the argument Lamb_par_numb, which is the number of parameters of the function Lambda(this is done for the
sake of generality(we don't know the parameters of the function Lambda in advance)).

Grad_Log_Lambda(x,Lamb_par): This function computes the gradient of the log intensity function of the poisson process for the function Lambda.
Notice that unlike Grad_Lambda we don't include Lamb_par_numb, this is likely an error and it might be a source of problems when generalizing the code.

gen_data(T,l,collection_input): This function generates samples from a OU process. 

cut(T,lmax,l,v): This function takes a sample path of a diffusion process and cuts it to the desired level of discretization given the origianl level.

IKBF(T,xin,lb,ls,U,obs,obs_time,dim,H,g_par): This function is the Irregular KBF, meaning that computes the KBF for a non-regular grid. It's used in the
test of the PF with Cox observations. 

KBF(T,l,lmax,z,collection_input): This function computes the KBF for a regular grid. It's used in the test of the PF with Cox observations. This and the previous 
function don't have a good description(check that)

ht(x,H, para=True): Function designed for the OU process. It computes the H@x.

Gox(obs,obs_times,x,Lambda,Lamb_par,l,N,dim,g_den,g_par) and G(): These functions compute the G function for the PF with Cox observations.
I didn't find much difference on them except that Gox of them is better commented and more polished. Notice that there is another function
G(), i.e. G(z,x,ht,H,l,para=True), this function is used in the test of the PF with Cox observations.

Gox_SM(obs,obs_times,x,Lambda,Grad_Log_Lambda,Lamb_par,Lamb_par_numb,\
    l,N,dim,g_den,Grad_Log_g_den,g_par): This function does the same as the previous one but additionally computes the gradient of the G function.

Gox_SM_W(obs,obs_times,x,x_nr,rc,Lambda,Grad_Log_Lambda,Lamb_par,Lamb_par_numb,\
    l,N,dim,g_den,Grad_Log_g_den,g_par): This function does the same as the previous, but solves and issue of the computation
meaning that it's not valid to use Gox_SM, or is it?

sr(W,N,x,dim): Function that uses systematic resampling (Not used for the PF with Cox observations)

multi_samp(W,N,x,dim): Multinomial sampling that relies on the numpy function random.choice. This function has complexity O(N logN) where
N is the number of particles (assuming that W has size N as in the PF). 

norm_logweights(lw,ax=0): This function takes the logweights and returns the normalized weights. It can be used in multiple dimensions thanks to ax=...

PF(T,z,lmax,x0,b_ou,A,Sig_ou,fi,ht,H,l,d,N,dim,resamp_coef,para=True): This function computes the PF for constant diffusion terms. 

Cox_PF(T,xin,b_ou,A,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par): This function computes the PF with Cox observations.

Cox_SM(T,xin,b_ou,A,Sig_ou,fi,Grad_b,b_numb_par,obs,obs_time,l,d,N,dim,resamp_coef,g_den,Grad_Log_g_den,g_par,Lambda,Grad_Lambda,Grad_Log_Lambda,\
    Lamb_par,Lamb_par_numb): This function computes the smoother for the PF with Cox observations(Is this wrong in comparison to the next function?)

Cox_SM_W(T,xin,b_ou,A,Sig_ou,fi,Grad_b,b_numb_par,obs,obs_time,l,d,N,dim,resamp_coef,g_den,Grad_Log_g_den,g_par,Lambda,Grad_Lambda,Grad_Log_Lambda,\
    Lamb_par,Lamb_par_numb):  This fucntion computes the smoother for the PF with Cox observations. It's the same as the previous one but solves an issue.

Cox_PE(T,xin,b_ou,A,Sig_ou,fi,Grad_b,b_numb_par,obs,obs_time,l,d,N,dim,resamp_coef,g_den,Grad_Log_g_den,g_par,Lambda,Grad_Lambda,Grad_Log_Lambda,\
Lamb_par,Lamb_par_numb,step_0,beta):
This function computes the parameter estimation for the PF with Cox observations.

"""

# %%

#%%
"""
def PF_ou(N,T,l,z,lz,collection_input): 
    [dim,dim_o,A,R1,R2,H,m0,C0]=collection_input
    # In the following lines we initialize the PF
    x0=np.random.multivariate_normal(m0,C0,N)
    J=T*(2**l)
    I=identity(dim).toarray()
    #print(l)
    tau=1./2**(l)
    L=la.expm(A*tau)
    ## We are going to need W to be symmetric! 
    W=(R1@R1)@(la.inv(A+A.T)@(L@(L.T)-I))
    for k in range(J):
    #in the following we propagate the particles using 
"""



def g_normal(x,g_pars):
    [dim,cov]=g_pars
    return x+np.random.multivariate_normal(np.zeros(dim),cov,len(x))

def gen_obs(x,l,T,Lambda,Lamb_par,g,g_par):
    # Function that generates poisson arrival times 
    # with intensity function Lambda(x)
    # ARGUMENTS: x: is a sample path of the difussion process, it is a rank 2
    # array with dimensions (T*2**l+1,dim), where dim is the dimension of the 
    # diff process.
    # l, T: are the time discretization level and T is the total time of the samples
    # Lamb: is a function that takes on two arguments, x and Lamb_par, which correspond
    # to the principal argument of the function and the hyperparameters of the 
    # function. 
    # g: is a function that generates the samples accordingly to a given kernel,
    # its arguments are the true position of the signal and the hyperparameters of 
    # the function.
    # OUTPUTS: We have 3 outputs 
    # cox_times: which is 
    # a rank 1 array with undetermined lenght.
    # ks: the nearest incexes of the cox_times according to the array x
    # obs: the observations generated, wich are a two rank array with dimension
    # (number of times sampled, dim)
    Gamma=np.cumsum(Lambda(x,Lamb_par,ax=1))/2**(l)
    #generate the times
    #print(Gamma)
    cox_times=[]
    c=1
    while c==1:
        U=np.random.uniform()
        s=-np.log(1-U)
        cou=Gamma-s
        #print(cou)
        sign=cou[0]
        k=0
   
        while sign<0 and c==1:
            k+=1
            if k>len(cou)-1:
                c=0
                k-=1
            sign=cou[k]
            
        #k is the first position such that Gamma[k]>s
        cox_times.append(k*2**(-l))
        
        #print(len(Gamma))
        Gamma=Gamma[k:]-Gamma[k]
        #print(t)
    cox_times= np.cumsum(np.array(cox_times)[:-1])
    ks=np.array(cox_times*2**(l),dtype=int)
    obs=g_normal(x[ks],g_par)
    
    return [cox_times,obs]


def gen_gen_data(T,x0,l,collection_input): #After generator of general data
    # parameters [dim,dim_o, b_ou,A,Sig_ou,fi,ht,H]
    # Function that generates euler maruyama samples of a difussion process
    # ARGUMENTS: T: final time of the discretization
    # x0: Initial position of the difussion, rank 1 array of dimension dim
    # l: level of discretization, the time step of this discretization is 2^{-l}
    # collection input: dim is the dimension of the difussion, b is a drift 
    # function that takes on two arguments, the diffusion array at an specific time
    # and the parameters of the functions A. The diffusion function of the process is 
    # Sig, which takes on the same first argument as b and the second argument is 
    # fi.
    # OUTPUTS: A rank two array of dimensions (T*2**l,dim) with the Euler-Maruyama
    # discretization.
    [dim, b,A,Sig,fi]=collection_input
    J=T*(2**l)
    I=identity(dim).toarray()
    #I_o=identity(dim_o).toarray()
    tau=2**(-l)
    v=np.zeros((J+1,dim))    
    #v[0]=np.random.multivariate_normal(m0,C0,(1)).T
    v[0]=x0



    for j in range(J):
        ## truth
        #print(np.shape(Sig(v[j],fi)),np.shape(b(v[j],A)))
        #print(b(v[j],A).shape,Sig(v[j],fi).shape)
        v[j+1] = v[j]+b(v[j],A)*tau + np.sqrt(tau)*(np.random.multivariate_normal(np.zeros(dim),I))@(Sig(v[j],fi).T)
        ## observation
    return v



#%%
def M(x0,b,A,Sig,fi,l,d,N,dim):
    # This is the function for the transition Kernel M(x,du): R^{d_x}->P(E_l)
    # ARGUMENTS: the argument of the Kenel x0 \in R^{d_x} (rank 1 dimesion dim=d_x array)
    # the drift and diffusion b, and Sig respectively (rank 1 and 2 numpy arrays of dim=d_x respectively)
    # the level of discretization l, the distance of resampling, the number of
    # particles N, and the dimension of the problem dim=d_x
    # OUTCOMES: x, an array of rank 3  2**l*d,N,dim that represents the path simuled
    # along the discretized time for a number of particles N.
    steps=int((np.float_power(2,l))*d)
    #print("this is the steps that is not working well",steps)
    dt=1./np.float_power(2,l)
    x=np.zeros((steps+1,N,dim))
    x[0]=x0
    I=identity(dim).toarray()
    for t in range(steps):
        dW=np.random.multivariate_normal(np.zeros(dim),I,N)*np.sqrt(dt)
        # Uncomment the following two lines for the GBM process
        #print("x shape is",x[t].shape)
        #print("Sig shape is",Sig(x[t],fi).shape)
        #diff=np.einsum("nd,njd->nj",dW,Sig(x[t],fi))
        #x[t+1]=x[t]+b(x[t],A)*dt+ diff
        # For the OU process comment the previous two lines and uncomment 
        # the following line
        #print(b(x[t],A).shape,Sig(x[t],fi).shape)
        x[t+1]=x[t]+b(x[t],A)*dt+ dW@(Sig(x[t],fi).T)
        # Uncomment the following lines for the nldt process
        #x[t+1]=x[t]+b(x[t],A)*dt+ dW*(Sig(x[t],fi))
    return x


def M_smooth(x0,b,A,Sig,fi,Grad_b,b_numb_par,l,d,N,dim):
    # This is the function for the transition Kernel M(x,du): R^{d_x}->P(E_l)
    # ARGUMENTS: the argument of the Kenel x0 \in R^{d_x} (rank 1 dimesion dim=d_x array)
    # the drift and diffusion b, and Sig respectively (rank 1 and 2 numpy arrays of dim=d_x respectively)
    # the level of discretization l, the distance of resampling, the number of
    # particles N, and the dimension of the problem dim=d_x
    # Grad_b is a function that takes (x,A) as argument and computes the gradnient of b wrt the 
    # parameters A, and evaluates it a (x,A).
    # b_numb_par is the number of parameters of the function b.
    # OUTCOMES: x, an array of rank 3  2**l*d,N,dim that represents the path simuled
    # along the discretized time for a number of particles N.
    # term0: an array of rank 3  N,dim,dim that represent a term in the computation of Lambda for 
    # the smoother for the first time step
    # term1: an array of rank 3  N,dim,dim that represent a term in the computation of Lambda for
    # the smoother for the rest of the time steps
    steps=int((2**(l))*d)
    dt=1./2**l
    x=np.zeros((steps+1,N,dim))
    x[0]=x0
    I=identity(dim).toarray()
    term=np.zeros((N,b_numb_par))
    term0=np.zeros((N,b_numb_par))
    #dWs=np.zeros((steps,N,dim))
    # Here we compute the first term of the gradient
    t=0
    dW=np.random.multivariate_normal(np.zeros(dim),I,N)*np.sqrt(dt)
    #dWs[0]=dW
    #print("dW_fist is ",dW)
    #print("dw is ",dW)
    # Uncomment the following two lines for the GBM and comment
    # the third line
    Sigma=Sig(x[t],fi)
    Sigma_inv=np.linalg.inv(Sigma)
    diff=np.einsum("nd,njd->nj",dW,Sigma)
    diff_2=np.einsum("nd,ndj->nj",dW,Sigma_inv)

    Gradient_b=Grad_b(x[t],A)
    term0=term0+np.einsum("nj,nij->ni",diff_2,Gradient_b)
    #print("x[t] is",x[t])
    x[t+1]=x[t]+b(x[t],A)*dt+ diff
    for t in np.array(range(steps))[1:]:
        dW=np.random.multivariate_normal(np.zeros(dim),I,N)*np.sqrt(dt)
        #dWs[t]=dW
        #print("dW_secodn is ",dW)
        # Uncomment the following two lines for the GBM and comment
        # the third line
        Sigma=Sig(x[t],fi)
        Sigma_inv=np.linalg.inv(Sigma)
        diff=np.einsum("nd,njd->nj",dW,Sigma)
        diff_2=np.einsum("nd,ndj->nj",dW,Sigma_inv)
        
        Gradient_b=Grad_b(x[t],A)
        term=term+np.einsum("nj,nij->ni",diff_2,Gradient_b)

        #print("x[t] is",x[t])
        x[t+1]=x[t]+b(x[t],A)*dt+diff
        # For the OU process comment the previous two lines and uncomment 
        # the following line
        #print(b(x[t],A).shape,Sig(x[t],fi).shape)
        #x[t+1]=x[t]+b(x[t],A)*dt+ dW@(Sig(x[t],fi).T)
        # Uncomment the following lines for the nldt process
        #x[t+1]=x[t]+b(x[t],A)*dt+ dW*(Sig(x[t],fi))
    return x,term0,term #,dWs




def M_smooth_W(x0,x_nr,b,A,Sig,fi,Grad_b,b_numb_par,l,d,N,dim):
    # This is the function for the transition Kernel M(x,du): R^{d_x}->P(E_l)
    # This function incorporates modification so it is helpful computing the smoother.
    # ARGUMENTS: the argument of the Kenel x0 \in R^{d_x} (rank 1 dimesion dim=d_x array)
    # the drift and diffusion b, and Sig respectively (rank 1 and 2 numpy arrays of dim=d_x respectively)
    # the level of discretization l, the distance of resampling, the number of
    # particles N, and the dimension of the problem dim=d_x
    # x_nr: the particles not-resampled particles. 
    # OUTCOMES: x, an array of rank 3  2**l*d,N,dim that represents the path simuled
    # along the discretized time for a number of particles N.
    # term0: an array of rank 3  N,dim,dim that represent a term in the computation of Lambda for 
    # the smoother for the first time step
    # term1: an array of rank 3  N,dim,dim that represent a term in the computation of Lambda for
    # the smoother for the rest of the time steps

    steps=int((2**(l))*d)
    dt=1./2**l
    x=np.zeros((steps+1,N,dim))
    x[0]=x0
    I=identity(dim).toarray()
    term=np.zeros((N,b_numb_par))
    term0=np.zeros((N,b_numb_par))
    #dWs=np.zeros((steps,N,dim))
    # Here we compute the first term of the gradient
    t=0
    dW=np.random.multivariate_normal(np.zeros(dim),I,N)*np.sqrt(dt)
    #dWs[0]=dW
    #print("dW_fist is ",dW)
    #print("dw is ",dW)
    # Uncomment the following two lines for the GBM and comment
    # the third line
    Sigma=Sig(x[t],fi)
    #print("Sigma is ",Sigma.shape)
    diff=np.einsum("nd,njd->nj",dW,Sigma)
    Sigma_nr=Sig(x_nr,fi)
    Sigma_inv_nr=np.linalg.inv(Sigma_nr)
    #print(Sigma_inv_nr.shape,(Sig(x[t],fi)).shape)
    diff_2_nr=np.einsum("nd,ndj->nj",dW,Sigma_inv_nr)
    #term0=term0+np.einsum("nj,ni->nji",diff_2_nr,x_nr)
    Gradient_b=Grad_b(x[t],A)
    term0=term0+np.einsum("nj,nij->ni",diff_2_nr,Gradient_b)
    #print("x[t] is",x[t])
    #print("diff shape is ",diff.shape)
    #print("x[t] shape is",x[t].shape)
    #print("A is",A)
    #print("b is", b(x[t],A).shape)
    #Uncomment the following line for the GBM 
    x[t+1]=x[t]+b(x[t],A)*dt+ diff
    #print("the shapes are", b(x[t],A).shape,(Sig(x[t],fi).T).shape,(dW@(Sig(x[t],fi).T)).shape)
    #x[t+1]=x[t]+b(x[t],A)*dt+ dW@(Sig(x[t],fi).T)
    for t in np.array(range(steps))[1:]:
        dW=np.random.multivariate_normal(np.zeros(dim),I,N)*np.sqrt(dt)
        #dWs[t]=dW
        #print("dW_secodn is ",dW)
        # Uncomment the following two lines for the GBM and comment
        # the third line
        Sigma=Sig(x[t],fi)
        Sigma_inv=np.linalg.inv(Sigma)
        diff=np.einsum("nd,njd->nj",dW,Sigma)
        diff_2=np.einsum("nd,ndj->nj",dW,Sigma_inv)
        Gradient_b=Grad_b(x[t],A)
        #if t==1:
            #print("x is ",x[t])
            #print("Gradient b is ",Gradient_b)
        term=term+np.einsum("nj,nij->ni",diff_2,Gradient_b)
        #term=term+np.einsum("nj,ni->nji",diff_2,x[t])
        #print("x[t] is",x[t])
        x[t+1]=x[t]+b(x[t],A)*dt+diff
        # For the OU process comment the previous two lines and uncomment 
        # the following line
        #print(b(x[t],A).shape,Sig(x[t],fi).shape)
        #x[t+1]=x[t]+b(x[t],A)*dt+ dW@(Sig(x[t],fi).T)
        # Uncomment the following lines for the nldt process
        #x[t+1]=x[t]+b(x[t],A)*dt+ dW*(Sig(x[t],fi))
    return x,term0,term #,dWs





#%%

#%%
#HERE WE TEST THE FUNCTION M_SMOOTH
"""
fi=np.array([[2]])
l=1
A=np.array([[1]])
d=1
N=3
dim=1
x0=np.random.normal(0.99,0.1,(N,dim))
print(x0)
print(M_smooth(x0,b_ou,A,Sig_ou,fi,l,d,N,dim))


# RESULTS: The dimensions check out.
"""
#%%


#%%

def b_gbm(x,mu):
    # Returns the drift "vector" evaluated at x
    # ARGUMENTS: x is a rank two array with dimensions where the first dimension 
    # corresponds to the number of particles and the second to the dimension
    # of the probelm. The second argument is mu, which is arank 1 array 
    # with dimension of the dimesion of the problem
    
    # OUTPUTS: A rank 2 array where the first dimension corresponds to the number
    # of particles and the second to the dimension of the system.
        #mu=reshape(mu,(-1,1))
        mult=x*mu
        #mult=np.array(mult)*10
        return mult

 
def Grad_b_gbm(x,mu):
    #this function is valid just for the one dimensional case
    # This function computes the gradient of the gbm drift 
    # ARGUMENTS: x is a rank two array with dimensions where the first dimension
    # corresponds to the number of particles and the second to the dimension
    # of the problem. 
    # mu is a rank 1 array with dimension of the dimesion of the problem
    # OUTPUTS: A rank 3 array with dimensions (N,1,dim) 
    return x[:,np.newaxis]




#We take the design of the brownian motion to the as the one in wikipedia:
#https://en.wikipedia.org/wiki/Geometric_Brownian_motion
    
def Sig_gbm(x,fi):
    # Returns the diffussion "vector" evaluated at x
    # ARGUMENTS: x is a rank two array with dimensions where the last dimension 
    # corresponds to the dimesion of the problem and the previous dimension 
    # (in case to exist) corresponds to the number of particles.
    # The second argument is fi, which is composed of a vector
    # sigs with dimension dim (of exclusively positive components)
    # and a square matrix Sig with the dimension of the 
    # system. The matrix Sig corresponds to the covariance matrix
    
    # OUTPUTS: A rank 3 array where the first dimension corresponds to the number
    # of particles and the second to the dimension of the system. Or a rank 3 array
    # 
    [sigs,Sig]=fi
    if x.ndim==1:
        Sig_m=((sigs*x)*Sig.T).T
    else:
        Sig_m=np.einsum("ij,ni->nij",Sig,sigs*x)
    return Sig_m



def Sig_nldt(x,fi):
    # Returns the diffussion "vector" evaluated at x
    # ARGUMENTS: x is a rank two array with dimensions where the last dimension 
    # corresponds to the dimesion of the problem and the previous dimension 
    # (in case to exist) corresponds to the number of particles.
    # The second argument is fi, which is composed of a vector
    # sigs with dimension dim (of exclusively positive components)
    # and a square matrix Sig with the dimension of the 
    # system. The matrix Sig corresponds to the covariance matrix
    
    # OUTPUTS: A rank 2 array where the first dimension and second dimension correspond
    # to the dimension of the diffusion. Or a rank 3 array where the first dimension
    # is the number of particles, the second and third are 1. In this case this problem is
    # only suitable for one dimensional problems. 
    
    if x.ndim==1:
        I=identity(x.shape[0]).toarray()    
        Sig_m=fi/np.sqrt(1+np.sum(x**2))*I
    
    # the following two lines work in the case we are using the regular
    # Particle filtering instead of the smoothing algorithm
    #else:
    #    Sig_m=fi/np.sqrt(1+np.sum(x**2,axis=1,keepdims=True))
    else:
        Sig_m=fi/np.sqrt(1+np.sum(x**2,axis=1,keepdims=True))
        Sig_m=Sig_m[:,np.newaxis,:]

    
    return Sig_m

def b_lan(x,df):
    # Returns the drift "vector" evaluated at x
    # ARGUMENTS: x is a rank two array with dimensions where the first dimension 
    # corresponds to the number of particles and the second to the dimension
    # of the probelm. 
    # The second argument is df, that stands for degrees of freedom and it is 
    # a positive scalar.
    
    # OUTPUTS: A rank 2 array where the first dimension corresponds to the number
    # of particles and the second to the dimension of the system.
        
        

        return -(df+1)*x/(df+x**2)
    


def b_ou(x,A):
    # Returns the drift "vector" evaluated at x
    # ARGUMENTS: x is a rank two array with dimensions where the first dimension 
    # corresponds to the number of particles and the second to the dimension
    # of the probelm. The second argument is A, which is a squared rank 2 array 
    # with dimension of the dimesion of the problem
    
    # OUTPUTS: A rank 2 array where the first dimension corresponds to the number
    # of particles and the second to the dimension of the system.
        
        mult=x@(A.T)
        #mult=np.array(mult)*10
        return mult


#dim=len(x.T)

def Grad_b_ou(x,A):
    #this function is made just for one dimension
    # This function computes the gradient of the langevin drift
    # ARGUMENTS: x is a rank two array with dimensions where the first dimension
    # corresponds to the number of particles and the second to the dimension
    # of the problem. 
    # df: is the degrees of freedom of the langevin equation.
    # OUTPUTS: A rank 3 array with dimensions (N,1,dim)
    return x[:,np.newaxis]
    
   


def Grad_b_lan(x,df):
    # This function computes the gradient of the langevin drift 
    # ARGUMENTS: x is a rank two array with dimensions where the first dimension
    # corresponds to the number of particles and the second to the dimension
    # of the problem. 
    # df: is the degrees of freedom of the langevin equation.
    # OUTPUTS: A rank 3 array with dimensions (N,1,dim) 
    term1=-x/(df+x**2)
    term2=(df+1)*x/(df+x**2)**2
    return (term1+term2)[:,np.newaxis,:]
#%%
def Sig_ou(x,fi):
    # Returns the Ornstein-Oulenbeck diffusion matrix 
    # ARGUMENTS: x is either a rank 2 array with dimensions (N,dim)
    # or a rank 1 array with dimension (dim) where N is the number of particles
    # and dim is the dimension of the problem
    # fi is a rank 2 array with dimensions (dim,dim)
    # OUTPUTS: A rank 3 array with dimensions (N,dim,dim) or (dim,dim) depending
    # on the rank of x.

    if x.ndim==1:
        return fi
    if x.ndim==2:
        fis=np.zeros((x.shape[0],x.shape[1],x.shape[1]))
        return fis+fi
# THE FOLLOWING FUNCTION IS REPLACED BY A VECTORIZED VERSION, IN ALL 
# TESTS PERFORMED THE BETTER TIMEWISE.
"""
def g_den(y,x,g_par):
    # Likelihood function
    # ARGUMENTS: 
    # y: as the observations, the function computes the conditional density of
    # y given x. y is a rank 2 array with dimensions (# of observations, dim).
    # x: position of the particles depending on the time, rank 3 array with 
    # dimensions (# of observations, particles, dim)
    # g_par: parameters of the density.
    # OUTPUT: 
    # gs: rank 2 array of the probabilities, with dimensions (# of observations,
    # # of particles)
    shape=x.shape
    #print(g_par)
    gs=np.zeros(shape[:2])
    for i in range(shape[0]):
        for n in range(shape[1]):
            var=multivariate_normal(mean=x[i,n],cov=g_par)
            gs[i,n]=var.pdf(y[i])
    return gs
"""

def g_den(y, x,   g_par  ):
    # Compute the conditional density of y given x
    # y: rank 2 array with dimensions (# of observations, dim)
    # x: rank 3 or rank 4 array with dimensions (# of observations, particles, dim)
    # (# of observations, particles, particles, dim)
    # g_par: parameters of the density (rank 2 array with dimensions (dim, dim))
    # Compute the mean of the multivariate normal distribution
    # OUTPUT: gs is a rank 3 or rank 4 depending on the rank of x with dimensions
    # (# of observations, particles) or (# of observations, particles, particles)
    # respectively.
    if x.ndim==3:
        mean = x
        # Compute the probabilities using the multivariate normal distribution and matrix multiplication
        diffs = y[:, np.newaxis, :] - mean
        exponent = -0.5 * np.sum(diffs @ np.linalg.inv(g_par) * (diffs), axis=2)
        gs = np.exp(exponent) / np.sqrt((2 * np.pi) ** mean.shape[-1] * np.linalg.det(g_par))


    if x.ndim==4:
        diffs=y[:,np.newaxis,np.newaxis,:]-x
        exponent = -0.5 * np.sum(np.dot(diffs, np.linalg.inv(g_par)) * (diffs), axis=-1)
        gs = np.exp(exponent) / np.sqrt((2 * np.pi) ** x.shape[-1] * np.linalg.det(g_par))
    return gs
    
#%%
def log_trans(xA,xB,b,A,Sig,fi,l):
    #This function computes the log of the transition density of the
    #Euler maruyama deiscretization at level l.
    #ARGUMENTS: xA: rank 2 array of the initial positions of the particles
    # and dimensions Nxdim
    #xB: rank 2 array of the final positions of the particles
    # and dimensions Nxdim
    #OUTPUT: rank two array with dimensions NxN.

    sigma=Sig(xA,fi)
    inv_sig=np.linalg.inv(sigma)
    mu=b(xA,A)/2**(l)+xA
    diff1=mu[:,np.newaxis]-xB
    diff2=np.einsum("ipk,ijk->ipj",diff1,inv_sig)
    diff3=-(1/2)*np.sum(diff2**2,axis=2)*2**l
    # here we include the 2*... bcs sigma is "standard deviation matrix" instead of the 
    # covariance matrix, since the cov matrix is sigma^T sigma then det(cov matrix)=det(sigma)^2
    log_det=2*np.log(np.linalg.det(sigma))- np.log(2)*((xA.shape)[1]*l)
    #note that the return does not include the log of the constant term
    # since it's not needed for the normalization of the weights.
    return diff3-(1/2)*log_det[:,np.newaxis]#-(1/2)*(x.shape)[1]*np.log(2*np.pi)

"""
# TEST FOR log_trans
dim=1
A=2*identity(dim).toarray()
fi=3*identity(dim).toarray()
xA=np.array([[1],[2],[3]])
xB=np.array([[0],[-2],[4]])
l=1
lt=log_trans(xA,xB,b_ou,A,Sig_ou,fi,l)
print(np.exp(lt))
# the results of the transition kernell are correct. 
# (note the to check this we included the constant term in the log_trans)
# we remove it since it's not necessary whenever we normalize the weights. 
"""

#%%
# TESting floor for g_den
"""
y=np.array([[1],[6]])
x=np.array([ [[[1],[2]],[[1],[3]]],[[[2],[3]],[[-1],[4]]]])
g=np.array([[2]])
print(g_den(y,x[:,0],g))
print(g_den(y,x,g))
# RESULTS: The dimensions check out and comparing the results with x[:,0] and x
# the results check out. So if  is correct  g_den(y,x[:,0],g) then g_den(y,x,g)
# is correct.
"""
#%%



def Grad_Log_g_den(y,x,g_par):
    # Gradient of the logarithm of the conditional density of y given x
    # y: rank 2 array with dimensions (# of observations, dim)
    # x: rank 3 array with dimensions (# of observations, particles, dim) or
    # rank 4 array with dimensions (# of observations, particles,particles, dim)
    # g_par: parameters of the density (rank 2 array with dimensions (dim, dim)), covariance matrix
    # Compute the mean of the multivariate normal distribution
    # The gradient of the log of the likelihood is computed as in
    # https://stats.stackexchange.com/questions/27436/how-to-take-derivative-of-multivariate-normal-density/276715#276715
    # (notice answer marked as correct is not actually correct)
    #OUTPUT: rank 4 array with dimensions (# of observations, particles, dim,dim)
    # or rank 5 array with dimensions (# of observations, particles,particles, dim,dim)
    if x.ndim==3:
        mean = x
        # Compute the probabilities using the multivariate normal distribution and matrix multiplication
        diffs = y[:, np.newaxis, :] - mean
        invSig=np.linalg.inv(g_par)
        term1=2*invSig
        np.fill_diagonal(term1,np.diagonal(invSig))
        term2=np.dot(diffs,invSig)
        term2= 2*np.einsum("ijl,ijk->ijlk",term2,term2)
        seq=np.array(range(x.shape[2]))
        term2[:,:,seq,seq]=term2[:,:,seq,seq]/2
        return -(1/2)*(term1-term2)
    if x.ndim==4:

        mean = x
        # Compute the probabilities using the multivariate normal distribution and matrix multiplication
        diffs = y[:, np.newaxis,np.newaxis, :] - mean

        invSig=np.linalg.inv(g_par)
        term1=2*invSig
        np.fill_diagonal(term1,np.diagonal(invSig))
        term2=np.dot(diffs,invSig)
        term2= 2*np.einsum("ijml,ijmk->ijmlk",term2,term2)
        seq=np.array(range(x.shape[-1]))
        term2[:,:,:,seq,seq]=term2[:,:,:,seq,seq]/2

        return -(1/2)*(term1-term2)

#TESTING FLOOR FOR Grad_Log_g_den
#%%
"""
# for dim=1
dim=2
y=np.array([[1],[6]])
x=np.array([[[[1,3],[2,.1]],[[1,3],[3,-6]]],[[[2,7],[3,0]],[[-1,3],[4,5]]]])
print(x.shape)
g_par=identity(dim).toarray()*2
print(Grad_Log_g_den(y,x,g_par).shape)
# for dim=2
dim=2
y=np.array([[1,2],[6,1]])
x=np.array([[[1,1],[2,1]],[[2,1],[3,1]]])
g_par=identity(dim).toarray()*2
print(Grad_Log_g_den(y,x,g_par))
"""
#%%

"""
AA=np.array([[[1,2],[3,4]],[[1,2],[3,4]]])
term1=2*AA
#np.fill_diagonal(term1,np.diagonal(AA))
#print(term1)
print(np.shape(AA[:,:,np.newaxis]))
print(np.shape(AA[np.newaxis,:,:]))
seq=np.array(range(2))
AA[:,seq,seq]=np.array([[1,2],[1,2]])
"""
#%%
def Lambda(x,Lamb_par,ax=0):
    # Intensity function of the Cox process. Appies a function on
    # x with the parameters Lamb_par along the axis ax
    # ARGUEMNTS:
    # x: Multirank array, ex: (# particles,dim)
    # Lamb_par: arguments feeded to the function
    # ax: is the axis of x to which we apply the function
    #Lamb=Lamb_par*np.linalg.norm(np.sin(x),axis=axis)**2
   # Lamb=Lamb_par*np.linalg.norm(np.sin(x),axis=axis)
    Lamb=Lamb_par*np.linalg.norm(x,axis=ax)
    
    return Lamb

def Grad_Lambda(x,Lamb_par,Lamb_par_numb,ax1=0):
    #THis function takes x and returns the gradient of the intensity function
    # Gradiend of the intensity function of the Cox process. Appies a function on
    # x with the parameters Lamb_par along the axis ax1
    # ARGUEMNTS:
    # x: Multirank array, ex: (time steps,# particles,dim)
    # Lamb_par: arguments feeded to the function
    # ax: is the axis of x to which we apply the function
    # OUTPUT: rank 2 array with dimensions (# of particles,# parameters=1)
    # NOTICE THAT WE USE KEEPDIMS=TRUE IN ORDER TO GIVE THE ADITIONAL DIMENSION # PARAMETERS=1
    # THIS ONLY WORKS IF AX1=-1
    return np.linalg.norm(x,axis=ax1,keepdims=True)
     


def Grad_Log_Lambda(x,Lamb_par):
    
    # Gradint of the log of the intensity function of the Cox process. Appies a function on
    # x with the parameters Lamb_par along the axis ax1
    # ARGUEMNTS:
    # x: Multirank array: either (# particles,dim) or (# particles,# particles,dim)
    # Lamb_par: arguments of the intensity function
    # OUTPUT: rank 2 array with dimensions either (# of particles,# parameters=1)
    # or (# of particles,# of particles,# parameters=1)
    if x.ndim==2:
        return np.zeros((x.shape[0],1))+1/Lamb_par
    if x.ndim==3:
        return np.zeros((x.shape[0],x.shape[1],1))+1/Lamb_par 




 #%%
"""
dim=1
y=np.array([[1]])
x=np.array([[[1],[2]]])
g_par=identity(dim).toarray()
print(g_den(y,x,g_par))
print(np.sqrt(1/(2*np.pi)))
"""
#%%
"""
x=np.array([[1,0],[0,1],[10,10]])



l=5

d=20

N=10
dim=10

x0=np.random.normal(1,1,dim)
np.random.seed(3)
comp_matrix = ortho_group.rvs(dim)
print(comp_matrix)
inv_mat=la.inv(comp_matrix)
S=diags(np.random.normal(1,0.1,dim),0).toarray()
fi=inv_mat@S@comp_matrix

B=diags(np.random.normal(-1,0.1,dim),0).toarray()*(2/3)
B=inv_mat@B@comp_matrix
#B=comp_matrix-comp_matrix.T  +B 
np.random.seed(3)
x3=M(x0,b_ou,B,Sig_ou,fi,l,d,N,dim)
"""


#Test for dimension 1
"""

x=np.array([[1,0],[0,1],[10,10]])
l=5
d=20
N=2
dim=1
x0=np.random.normal(1,1,dim)
np.random.seed(3)
comp_matrix=[[1]]

inv_mat=la.inv(comp_matrix)
S=diags(np.random.normal(1,0.1,dim),0).toarray()
fi=inv_mat@S@comp_matrix
B=diags(np.random.normal(-1,0.1,dim),0).toarray()*(2/3)
B=inv_mat@B@comp_matrix
#B=comp_matrix-comp_matrix.T  +B 
print(B)
np.random.seed(3)
x3=M(x0,b_ou,B,Sig_ou,fi,l,d,N,dim)
print(x3)



"""

#%%

"""
l=5
d=1
N=20
T=10
dim=5
dim_o=dim
#x=M(x0,b_ou,Sig_ou,l,d,N,dim)
I=identity(dim).toarray()
I_o=identity(dim_o).toarray()
#R2=(identity(dim_o).toarray() + np.tri(dim_o,dim_o,1) - np.tri(dim_o,dim_o,-2))/20
R2=I
H=rand(dim_o,dim,density=0.75).toarray()/1e-1

x0=np.random.normal(1,0,dim)
np.random.seed(3)
comp_matrix = ortho_group.rvs(dim)
inv_mat=la.inv(comp_matrix)
S=diags(np.random.normal(1,0.001,dim),0).toarray()
fi=inv_mat@S@comp_matrix

B=diags(np.random.normal(-.1,0.001,dim),0).toarray()*(2/3)
A=inv_mat@B@comp_matrix

#A=b_ou(I,B).T
R1=Sig_ou(np.zeros(dim),fi)
x= M(x0,b_ou,A,Sig_ou,fi,l,d,N,dim)

print(x.shape)
times=np.array(range(int(2**l*d+1)))/2**l
#plt.plot(times,x2[:,7,0])
plt.plot(times,x[:,0,0])
"""
#%%
# classical data generator from the Unbiased EnKBF project.
def gen_data(T,l,collection_input):
    [dim,dim_o,A,R1,R2,H,m0,C0]=collection_input
    J=T*(2**l)
    I=identity(dim).toarray()
    tau=2**(-l)
    L=la.expm(A*tau)
    ## We are going to need W to be symmetric! 
    W=(R1@R1)@(la.inv(A+A.T)@(L@(L.T)-I))
    W=(W+W.T)/2.
    C=tau*H
    V=(R2@R2)*tau

    v=np.zeros((J+1,dim,1))
    z=np.zeros((J+1,dim_o,1))
    #v[0]=np.random.multivariate_normal(m0,C0,(1)).T
    v[0]=np.random.multivariate_normal(m0,C0,(1)).T
    z[0]=np.zeros((dim_o,1))


    for j in range(J):
        ## truth
        v[j+1] = L@v[j] + np.random.multivariate_normal(np.zeros(dim),W,(1)).T
        ## observation
        z[j+1] = z[j] + C@v[j+1] + np.random.multivariate_normal(np.zeros(dim_o),V,(1)).T
        
    return([z,v])

# classical cutting function from the Unbiased EnKBF project.

def cut(T,lmax,l,v):
    #chooses a grid of data along the first axis of v
    ind = np.arange(T*2**l+1)
    rtau = 2**(lmax-l)
    w = v[ind*rtau]
    return(w)




def IKBF(T,xin,lb,ls,U,obs,obs_time,dim,H,g_par): 

    # The I in front of KBF stands for irregular
    # This function only considers observations in the same dimensions as the 
    # signal process.
    lo=len(obs) 
    # lo stands for lenght of observations 
    pt=0#previous time
    P=identity(dim).toarray()
    xf=np.zeros((lo+1,dim))
    xf[0]=xin
    for i in range(lo):
        ct=obs_time[i] #current time
        Fk=((U.T)*np.exp(lb*(ct-pt)))@(U)
        xh=Fk@xf[i]
        Qk=Fk@(1/2)*((U.T)*(ls**2*lb*(np.exp(lb*(ct-pt)-1))))@U
        Phk=Fk@P(Fk.T)+Qk
        dk=obs[i]-H@xh
        Sk=H@Phk@(H.T)+g_par
        Kk=Phk@(H.T)@np.linalg.inv(Sk)
        xf[i+1]=xh+Kk@dk
        
    return xf

def KBF(T,l,lmax,z,collection_input):
    
    [dim,dim_o,A,R1,R2,H,m0,C0]=collection_input
    J=T*(2**l)
    I=identity(dim).toarray()
    tau=2**(-l)
    L=la.expm(A*tau)
    W=(R1@R1)@(la.inv(A+A.T)@(L@(L.T)-I))
    W=(W+W.T)/2.
    
    ## C: dim_o*dim matrix
    C=tau*H
    V=(R2@R2)*tau
    
    z=cut(T,lmax,l,z)
    m=np.zeros((J+1,dim,1))
    c=np.zeros((J+1,dim,dim))
    m[0]=np.array([m0]).T
    c[0]=C0
    
    for j in range(J):
       
        ## prediction mean-dim*1 vector
        mhat=L@m[j]
        ## prediction covariance-dim*dim matrix
        chat=L@c[j]@(L.T)+W
        ## innovation-dim_o*1 vector
        d=(z[j+1]-z[j])-C@mhat
        ## Kalman gain-dim*dim_o vector
        K=(chat@(C.T))@la.inv(C@chat@(C.T)+V)
        ## update mean-dim*1 vector
        
        m[j+1]=mhat+K@d
        ## update covariance-dim*dim matrix
        c[j+1]=(I-K@C)@chat
    return([m,c])

"""
"""

#%%
"""
l=8
d=10
N=10
T=10
dim=5
dim_o=dim
#x=M(x0,b_ou,Sig_ou,l,d,N,dim)


#R2=(identity(dim_o).toarray() + np.tri(dim_o,dim_o,1) - np.tri(dim_o,dim_o,-2))/20
H=rand(dim_o,dim,density=0.75).toarray()/1e-1
I=identity(dim).toarray()
R2=I
np.random.seed(3)
comp_matrix = ortho_group.rvs(dim)
inv_mat=la.inv(comp_matrix)
#A=b_ou(I,B).T
S=diags(np.random.normal(1,0.1,dim),0).toarray()
fi=inv_mat@S@comp_matrix

B=diags(np.random.normal(-.1,0.1,dim),0).toarray()*(2/3)
A=inv_mat@B@comp_matrix
R1=Sig_ou(np.zeros(dim),fi)
np.random.seed(2)
C0=I*1e-3
m0=np.random.multivariate_normal(np.zeros(dim),I)
collection_input=[dim,dim_o,A,R1,R2,H,m0,C0]

    
[z,x_true]=gen_data(T,l,collection_input)
#z=np.reshape(z,z.shape[:2])


"""
#test for dimension 1
"""
l=8
d=10
N=3
T=10
dim=1
dim_o=dim
#x=M(x0,b_ou,Sig_ou,l,d,N,dim)


#R2=(identity(dim_o).toarray() + np.tri(dim_o,dim_o,1) - np.tri(dim_o,dim_o,-2))/20
H=rand(dim_o,dim,density=0.75).toarray()/1e-1
I=identity(dim).toarray()
R2=I
np.random.seed(3)
#comp_matrix = ortho_group.rvs(dim)
comp_matrix=[[1]]
inv_mat=la.inv(comp_matrix)
#A=b_ou(I,B).T
S=diags(np.random.normal(1,0.1,dim),0).toarray()
fi=inv_mat@S@comp_matrix
B=diags(np.random.normal(-.1,0.1,dim),0).toarray()*(2/3)
A=inv_mat@B@comp_matrix
R1=Sig_ou(np.zeros(dim),fi)
np.random.seed(2)
C0=I*1e-3
m0=np.random.multivariate_normal(np.zeros(dim),I)
collection_input=[dim,dim_o,A,R1,R2,H,m0,C0]
#print(collection_input)
[z,x_true]=gen_data(T,l,collection_input)
#z=np.reshape(z,z.shape[:2])

#print(z)
"""


#%%

"""
lmax=l
kbf=KBF(T,l,lmax,z,collection_input)
plt.plot(2**(-l)*np.array(range(T*2**l+1)),x_true[:,0,0],label="Signal")
plt.plot(2**(-l)*np.array(range(T*2**l+1)),kbf[0][:,0,0],label="KBF")



lmax=8
l=lmax
d=10
np.random.seed(11)
x_star=M(x0,b_ou,B,Sig_ou,fi,l,d,N,dim)
print(x_star[-1])
"""
#%%
#%%
#[m,c]=KBF(T,l,lmax,z,collection_input)
def ht(x,H, para=True):
    #This function takes as argument a rank 3 array where for each element(2 rank array)
    # x[i]  the function applies x[i]@H.T
    #ARGUMENTS: rank 3 array x, para=True computes the code with the einsum 
    #function from numpy. Otherwise the function is computed with a for
    #in the time discretization
    #OUTPUTS: rank 3 array h
    if para==True:
        h=np.einsum("ij,tkj->tki",H,x)
    else:
        h=np.zeros(x.shape)
        for i in range(len(x)):
            h[i]=x[i]@(H.T)
    return h
            


def G(obs,obs_times,x,Lambda,Lamb_par,l,N,dim,g_den,g_par):
    #Function that computes the weights of the system depending on the observations
    
    #ARGUMENTS: obs: are the observations(2 rank array) with dimensions (# of observations,dim)
    # x: is the array of the 
    # particles (rank 3 array) with dimensions (2**l+1,N,dim), where N is the number
    # of prticles.
    # obs_times: are the times of observation (cox process).
    # Lambda: intensity function of the poisson process, takes on two arguments,
    # x as the principal argumetn and Lamb_par as hyperparameters.
    # g_den: likelihood function of the observation once the observation time
    # has been computed, it has three parameters, 
    #OUTPUT: logarithm of the weights 
    log_int=-np.sum(Lambda(x[1:],Lamb_par,ax=2),axis=0)/2**(l)
    #log_int=-np.sum(Lambda(x[:-1],Lamb_par,axis=2))/2**(l)
    x_inter=np.zeros((len(obs),N,dim))
    
    if len(obs_times)>0:
        
        for i in range(len(obs)):
            t_obs=obs_times[i]
            k=int(np.floor(t_obs*2**l))
            x_inter[i]=(x[k+1]-x[k])*(t_obs-(k+1)/2**(l))*2**(l)+x[k+1]
            
        log_prod1=np.sum(np.log(Lambda(x_inter,Lamb_par,ax=2)),axis=0)
        log_prod2=np.sum(np.log(g_den(obs,x_inter,g_par)),axis=0)
    else:
        log_prod1=0
        log_prod2=0
        
    #print(suma1,suma2)
    log_w=log_int+log_prod1+log_prod2
   
    return log_w

#%%
#Gox tests 

"""
obs=np.array([[1],[2]])
obs_times=np.array([0.25/2,0.75+0.25/2])
x=np.array([[[1],[2]],[[7],[2]],[[2],[3]]])
Lamb_par=2
l=1
N=2
dim=1
g_par=identity(dim).toarray()*3
print(Gox(obs,obs_times,x,Norm_Lambda,Lamb_par,l,N,dim,g_den,g_par))
print(x.shape)
"""
#%%

def Gox(obs,obs_times,x,Lambda,Lamb_par,l,N,dim,g_den,g_par):
    #Function that computes the logs of the weights of the system depending on the observations
    #ARGUMENTS: obs: are the observations(2 rank array) with dimensions (# of observations,dim)
    # x: is the array of the 
    # particles (rank 3 array) with dimensions (2**l+1,N,dim), where N is the number
    # of prticles.
    # obs_times: are the times of observation (cox process). Note tha these observations
    # start with time 0, meaning that if Gox is used for a iteration of shifted interval
    # then the obs_times must be shifted accordingly.
    # Lambda: intensity function of the poisson process, takes on two arguments,
    # x as the principal argumetn and Lamb_par as hyperparameters.
    # g_den: likelihood function of the observation once the observation time
    # has been computed, it has three parameters, 
    #OUTPUT: logarithm of the weights, rank 1 dimention N
    log_int=-np.sum(Lambda(x[:-1],Lamb_par,ax=2),axis=0)/np.float_power(2,l)
    #print(Lambda(x[1:],Lamb_par,ax=2))
    x_inter=np.zeros((len(obs),N,dim)) 
    if len(obs_times)>0:
        for i in range(len(obs)):
            t_obs=obs_times[i]
            #k=int(np.floor(t_obs*2**l))
            k=int(np.ceil(t_obs*np.float_power(2,l)))
            #print("k is",k)
            #print("x shape is",x.shape)
            #print(k,t_obs*2**l)
            #Interpolation of the diffusion given an observation time 
            x_inter[i]=(x[k]-x[k-1])*(t_obs-(k-1)/np.float_power(2,l))*np.float_power(2,l)+x[k-1]
            #x_inter[i]=(x[k+1]-x[k])*(t_obs-(k+1)/2**(l))*2**(l)+x[k+1]
            #print(x_inter[i],x[k])

        #print("x_inter for l=",l," are ",x_inter)
        log_prod1=np.sum(np.log(Lambda(x_inter,Lamb_par,ax=2)),axis=0)
        log_prod2=np.sum(np.log(g_den(obs,x_inter,g_par)),axis=0)
        #print(log_prod1.shape,log_prod2.shape,log_int.shape)
    else:
        log_prod1=0
        log_prod2=0
    #print(log_prod1,log_prod2,log_int)
    #print(suma1,suma2)
    log_w=log_int+log_prod1+log_prod2
    return log_w   
#%%
"""
import numpy as np

# Example array
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Threshold value
threshold = 5

# Split the array into two based on the threshold
arr1, arr2 = np.split(arr, np.where(arr >= threshold)[0][0:1])

print(arr1)
# Output: [1 2 3 4 5]

print(arr2)
# Output: [6 7 8 9 10]

"""

#%%
"""
A=np.array([0,2,3,5])
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Threshold value
threshold = 0

# Find the index where the elements of arr exceed the threshold
split_index = np.argmax(arr >= threshold)
print(split_index)
if split_index==0 and arr[0]<threshold:
    split_index=len(arr)

# Split the array into two based on the threshold
arr1 = arr[:split_index]
arr2 = arr[split_index:]
print(arr1,arr2)

"""
#%%
#%%

def Gox_SM(obs,obs_times,x,Lambda,Grad_Log_Lambda,Lamb_par,Lamb_par_numb,\
    l,N,dim,g_den,Grad_Log_g_den,g_par):
    #Function that computes the logs of the weights of the system depending on the observations,
    # additionally it also computes the gradients of the log of the intensity function and the
    # gradients of the log of the likelihood function, finally it also computes log_w_init, which is necessary
    # for the computation of the smoothing in some cases. 
    #ARGUMENTS: obs: are the observations(2 rank array) with dimensions (# of observations,dim)
    # x: is the array of the 
    # particles (rank 3 array) with dimensions (2**l+1,N,dim), where N is the number
    # of prticles.
    # obs_times: are the times of observation (cox process).
    # Lambda: intensity function of the poisson process, takes on two arguments,
    # x as the principal argumetn and Lamb_par as hyperparameters.
    # g_den: likelihood function of the observation once the observation time
    # has been computed, it has three parameters, 
    # Grad_Log_Lambda: Gradient of the log of the intensity function of the Cox process. 
    # it has two argumetns, x, and Lamb_par
    # Lamb_par_numb: number of parameters of the intensity function
    # Grad_Log_g_den: Gradient of the log of the likelihood function of the observations.
    # it has three argumetns, y, x, and g_par
    # g_par_numb: number of parameters of the likelihood function
    #OUTPUT: logarithm of the weights, rank 1 dimention N
    # Lamb_par_grad: sum of the gradients of the logarithms of the intensity function for all
    # the observations, the dimensions are either (N,Lamb_par_numb) or (N,N,Lamb_par_numb)
    # depending on whether there are observations before the first time step.
    # g_par_grad: sum of the gradients of the logarithms of the likelihood function for all
    # the observations, the dimensions are (N,dim,dim  ) or (N,N,dim,dim) depending on whether
    # there are observations before the first time step.
    # log_w_init: logarithm of the weights for the first time step, depending on the 
    # wheter we have observations before the first time step, the dimensions are either
    # (N) or (N,N).
    log_int=-np.sum(Lambda(x[:-1],Lamb_par,ax=2),axis=0)/2**(l)
    log_int_init= -np.sum(Lambda(x[:1],Lamb_par,ax=2),axis=0)/2**(l)

    #print("log_int_init is",log_int_init)
    #print(Lambda(x[1:],Lamb_par,ax=2))
    x_inter=np.zeros((len(obs),N,dim))
    # IN THE FOLLOWING WE COMPUTE THE GRADIENTS OF THE LOG OF THE INTENSITY FUNCTION FOR TWO CASES
    # THE FIRST ONE IS THE ONE IN WHICH THE INTERPOLATION OF THE SIGNAL X_INTER DOES  DEPEND ON x[0],
    # THUS WE HAVE TO COMPUTE THE GRADIENTS FOR A N*N COMBINATION OF PARTICLES
    # THE SECOND CASE IS THE ONE IN WHICH THE INTERPOLATION OF THE SIGNAL X_INTER DOES NOT DEPEND ON x[0],
    # THUS WE HAVE TO COMPUTE THE GRADIENTS FOR A N COMBINATION OF PARTICLES 
    if len(obs_times)>0:

        threshold=2**(-l)
        split_index = np.argmax(obs_times >= threshold)
        if split_index==0 and obs_times[0]<threshold:
            split_index=len(obs_times)
        # Split the array into two based on the threshold
        mixed_var_times = obs_times[:split_index]
        single_var_times =obs_times[split_index:]
        if len(mixed_var_times)>0:
            Lamb_par_grad=np.zeros((N,N,Lamb_par_numb))
            g_par_grad=np.zeros((N,N,dim,dim))

        else:
            Lamb_par_grad=np.zeros((N,Lamb_par_numb))
            g_par_grad=np.zeros((N,dim,dim))

        if len(mixed_var_times)>0:
            x_inter_mixed=np.zeros((len(mixed_var_times),N,N,dim))
            seq=np.array(range(N))
            for i in range(len(mixed_var_times)):
                t_obs=obs_times[i]
                k=int(np.ceil(t_obs*2**l))
                # notice that doing this the axis 1(starting from 0) of x_inter_mixed corresponds to the u_{i-1} and 
                # the axis 2(starting from 0) of x_inter_mixed corresponds to the u_{i}.
                x_inter_mixed[i]=(x[k,np.newaxis]-x[k-1,:,np.newaxis])*(t_obs-(k-1)/2**(l))*2**(l)+x[k-1,:,np.newaxis]
                x_inter[i]=x_inter_mixed[i,seq,seq]
                Lamb_par_grad=Lamb_par_grad+Grad_Log_Lambda\
                (x_inter_mixed[i],Lamb_par)
            g_par_grad=np.sum(Grad_Log_g_den(obs[:len(mixed_var_times)],x_inter_mixed,g_par),axis=0)
            #print("term3A is",g_par_grad   )
            #print("x_mix_inter is",x_inter_mixed)
            #print("grad_log_g is",( -1/2)*(1/2-1/4*((x_inter_mixed-1)**2)))
            log_prod1_init=np.sum(np.log(Lambda(x_inter_mixed,Lamb_par,ax=-1)),axis=0)
            log_prod2_init=np.sum(np.log(g_den(obs[:len(mixed_var_times)],x_inter_mixed,g_par)),axis=0)
            #print("log_prod2_init is",log_prod2_init)
            log_w_init=log_int_init[:,np.newaxis]+log_prod1_init+log_prod2_init
            #print("x_i_m is",x_inter_mixed)
            #print("x is ",x)
        else:
            log_w_init=log_int_init

        for i in range(len(single_var_times)):
            i=i+len(mixed_var_times)
            t_obs=obs_times[i]
            k=int(np.ceil(t_obs*2**l))
            #print(k,t_obs*2**l)
            #Interpolation of the diffusion given an observation time 
            x_inter[i]=(x[k]-x[k-1])*(t_obs-(k-1)/2**(l))*2**(l)+x[k-1]
            Lamb_par_grad=Lamb_par_grad+Grad_Log_Lambda\
            (x_inter[i],Lamb_par)
            #here we can just add the gradient since there are two scenariosn, in the first one
            # len(mixed_var_times)>0 and the arrays will broadcast(the dimensions match the variables u_{i-1} and u_{i})
            # , and in the second one len(mixed_var_times)=0 and the g_par_grad is a N*dim*dim array equal to zero
        g_par_grad=g_par_grad+ np.sum(Grad_Log_g_den(obs[len(mixed_var_times):],x_inter[len(mixed_var_times):],g_par),axis=0)
        #print("x_inter",x_inter)
        #print("term3B is",np.sum(Grad_Log_g_den(obs[len(mixed_var_times):],x_inter[len(mixed_var_times):],g_par),axis=0))
        log_prod1=np.sum(np.log(Lambda(x_inter,Lamb_par,ax=2)),axis=0)
        log_prod2=np.sum(np.log(g_den(obs,x_inter,g_par)),axis=0)
        #print(log_prod1.shape,log_prod2.shape,log_int.shape)
    else:
        log_prod1=0
        log_prod2=0
        log_w_init=log_int_init
        Lamb_par_grad=0
        g_par_grad=0
    #print(log_prod1,log_prod2,log_int)
    #print(suma1,suma2)
        
        
   
    #print("x_inter is",x_inter)

    log_w=log_int+log_prod1+log_prod2
    return log_w ,log_w_init, Lamb_par_grad,g_par_grad


def Gox_SM_W(obs,obs_times,x,x_nr,rc,Lambda,Grad_Log_Lambda,Lamb_par,Lamb_par_numb,\
    l,N,dim,g_den,Grad_Log_g_den,g_par):

    #Function that computes the logs of the weights of the system depending on the observations
    # this function is adapted for the smoother in two ways, the first one, it is changed to compute
    # additional wegihts, also it computes the gradient of the log of the intensity function and the
    # gradient of the log of the likelihood of the observations.
    # Additionally to the previous changes, this function changes wrt Gox_SM adding the 
    # arguments x_nr and rc, this way computing different weights(log_w_init) and gradients
    # for the not resampled particles.
    #ARGUMENTS: obs: are the observations(2 rank array) with dimensions (# of observations,dim)
    # x: is the array of the 
    # particles (rank 3 array) with dimensions (2**l+1,N,dim), where N is the number
    # of prticles.
    # x_nr: is the array of particles corresponding to the non-resampled particles its dimensions are
    # (N,dim)
    # rc: is the array of resampling coefficients, its dimensions are (N)
    # obs_times: are the times of observation (cox process).
    # Lambda: intensity function of the poisson process, takes on two arguments,
    # x as the principal argumetn and Lamb_par as hyperparameters.
    # g_den: likelihood function of the observation once the observation time
    # has been computed, it has three parameters, 
    # Grad_Log_Lambda: Gradient of the log of the intensity function of the Cox process. 
    # it has two argumetns, x, and Lamb_par
    # Lamb_par_numb: number of parameters of the intensity function
    # Grad_Log_g_den: Gradient of the log of the likelihood function of the observations.
    # it has three argumetns, y, x, and g_par
    # g_par_numb: number of parameters of the likelihood function
    #OUTPUT: logarithm of the weights, rank 1 dimention N
    # Lamb_par_grad: sum of the gradients of the logarithms of the intensity function for all
    # the observations, the dimensions are either (N,Lamb_par_numb) or (N,N,Lamb_par_numb)
    # depending on whether there are observations before the first time step.
    # g_par_grad: sum of the gradients of the logarithms of the likelihood function for all
    # the observations, the dimensions are (N,dim,dim  ) or (N,N,dim,dim) depending on whether
    # there are observations before the first time step.
    # log_w_init: logarithm of the weights for the first time step, depending on the 
    # wheter we have observations before the first time step, the dimensions are either
    # (N) or (N,N). 
    log_int=-np.sum(Lambda(x[:-1],Lamb_par,ax=2),axis=0)/2**(l)
    log_int_init= -np.sum(Lambda(x_nr[np.newaxis,:],Lamb_par,ax=2),axis=0)/2**(l)
    x_orig=x[0].copy()
    #print("log_int_init is",log_int_init)
    #print(Lambda(x[1:],Lamb_par,ax=2))
    x_inter=np.zeros((len(obs),N,dim))
    # IN THE FOLLOWING WE COMPUTE THE GRADIENTS OF THE LOG OF THE INTENSITY FUNCTION FOR TWO CASES
    # THE FIRST ONE IS THE ONE IN WHICH THE INTERPOLATION OF THE SIGNAL X_INTER DOES  DEPEND ON x[0],
    # THUS WE HAVE TO COMPUTE THE GRADIENTS FOR A N*N COMBINATION OF PARTICLES
    # THE SECOND CASE IS THE ONE IN WHICH THE INTERPOLATION OF THE SIGNAL X_INTER DOES NOT DEPEND ON x[0],
    # THUS WE HAVE TO COMPUTE THE GRADIENTS FOR A N COMBINATION OF PARTICLES 
    if len(obs_times)>0:
        x[0]=x_nr # here we change the initial value of the particles to the non-resampled ones, notice that
        # since x is a mutable object then we are changing the value of x[0] in the function Gox_SM, this doesn't
        #matter much since we don't use the x[0] any other time in the functiond
        threshold=2**(-l)
        split_index = np.argmax(obs_times >= threshold)
        if split_index==0 and obs_times[0]<threshold:
            split_index=len(obs_times)
        # Split the array into two based on the threshold
        mixed_var_times = obs_times[:split_index]
        single_var_times =obs_times[split_index:]
        if len(mixed_var_times)>0:
            Lamb_par_grad=np.zeros((N,N,Lamb_par_numb))
            g_par_grad=np.zeros((N,N,dim,dim))

        else:
            Lamb_par_grad=np.zeros((N,Lamb_par_numb))
            g_par_grad=np.zeros((N,dim,dim))

        if len(mixed_var_times)>0:
            x_inter_mixed=np.zeros((len(mixed_var_times),N,N,dim))
            seq=np.array(range(N))
            for i in range(len(mixed_var_times)):
                t_obs=obs_times[i]
                k=int(np.ceil(t_obs*2**l))
                # notice that doing this the axis 1(starting from 0) of x_inter_mixed corresponds to the u_{i-1} and 
                # the axis 2(starting from 0) of x_inter_mixed corresponds to the u_{i}.
                x_inter_mixed[i]=(x[k,np.newaxis]-x[k-1,:,np.newaxis])*(t_obs-(k-1)/2**(l))*2**(l)+x[k-1,:,np.newaxis]
                # the only way in which log_w changes depending on the resampling is through the interpolation,
                # we change it so it is aligned with the resampled particles.
                x_inter[i]=x_inter_mixed[i,rc,seq]
                Lamb_par_grad=Lamb_par_grad+Grad_Log_Lambda\
                (x_inter_mixed[i],Lamb_par)
            g_par_grad=np.sum(Grad_Log_g_den(obs[:len(mixed_var_times)],x_inter_mixed,g_par),axis=0)
            #print("term3A is",g_par_grad   )
            #print("x_mix_inter is",x_inter_mixed)
            #print("grad_log_g is",( -1/2)*(1/2-1/4*((x_inter_mixed-1)**2)))
            log_prod1_init=np.sum(np.log(Lambda(x_inter_mixed,Lamb_par,ax=-1)),axis=0)
            log_prod2_init=np.sum(np.log(g_den(obs[:len(mixed_var_times)],x_inter_mixed,g_par)),axis=0)
            #print("log_prod2_init is",log_prod2_init)
            log_w_init=log_int_init[:,np.newaxis]+log_prod1_init+log_prod2_init
            #print("x_i_m is",x_inter_mixed)
            #print("x is ",x)
        else:
            log_w_init=log_int_init

        for i in range(len(single_var_times)):
            i=i+len(mixed_var_times)
            t_obs=obs_times[i]
            k=int(np.ceil(t_obs*2**l))
            #print(k,t_obs*2**l)
            #Interpolation of the diffusion given an observation time 
            x_inter[i]=(x[k]-x[k-1])*(t_obs-(k-1)/2**(l))*2**(l)+x[k-1]
            Lamb_par_grad=Lamb_par_grad+Grad_Log_Lambda\
            (x_inter[i],Lamb_par)
            #here we can just add the gradient since there are two scenariosn, in the first one
            # len(mixed_var_times)>0 and the arrays will broadcast(the dimensions match the variables u_{i-1} and u_{i})
            # , and in the second one len(mixed_var_times)=0 and the g_par_grad is a N*dim*dim array equal to zero
        g_par_grad=g_par_grad+ np.sum(Grad_Log_g_den(obs[len(mixed_var_times):],x_inter[len(mixed_var_times):],g_par),axis=0)
        #print("x_inter",x_inter)
        #print("term3B is",np.sum(Grad_Log_g_den(obs[len(mixed_var_times):],x_inter[len(mixed_var_times):],g_par),axis=0))
        log_prod1=np.sum(np.log(Lambda(x_inter,Lamb_par,ax=2)),axis=0)
        log_prod2=np.sum(np.log(g_den(obs,x_inter,g_par)),axis=0)
        #print(log_prod1.shape,log_prod2.shape,log_int.shape)
    else:
        log_prod1=0
        log_prod2=0
        log_w_init=log_int_init
        Lamb_par_grad=0
        g_par_grad=0
    #print(log_prod1,log_prod2,log_int)
    #print(suma1,suma2)
        
        
   
    #print("x_inter is",x_inter)
    x[0]=x_orig
    log_w=log_int+log_prod1+log_prod2
    return log_w ,log_w_init, Lamb_par_grad,g_par_grad  



#How should we test Gox_SM? 
#Comparation with actual computations? 
#test lambda
#test computations for different parameters 


#TESTING FLOOR FOR Gox_SM

#No observations:
"""
dim=1
Lamb_par_numb=1
N=2
obs=np.array([[0.6],[0.64]])
obs_times=np.array([.3,0.4])
x=np.array([[[0.1],[0.3]],[[0.15],[0.5]],[[0.6],[0.7]]])
l=1
Lamb_par=3

g_par=identity(dim).toarray()
x_int=np.array([[[0.14],[0.46]],[[0.15],[0.5]]])
#print(np.sum(Grad_Log_g_den(obs,x_int,g_par),axis=0))
print(Gox_SM(obs,obs_times,x,Lambda,Grad_Log_Lambda,Lamb_par,Lamb_par_numb,\
l,N,dim,g_den,Grad_Log_g_den,g_par)[1])
#print(np.log(np.sqrt(1/(np.pi*2))*np.exp(-1*(0.15-0.5)**2/2)))

# RESULTS

# the mixed interpolation with x_int=np.array([[[0.14],[0.46]],[[0.15],[0.5]]])
# and obs_times=np.array([.25,0.5]) is [[[[0.125][0.3  ]][[0.225][0.4  ]]]], both
# the values and dimensions are correct.


# The dimension for the gradient of the logs are correct when implementing the function
# with observations before the first time step, the dimensions are (N,N,Lamb_par_numb) and
# (N,N,dim,dim) for the intensity and likelihood functions respectively.
#"""
#%%
#How should we test Gox? 
#Comparation with actual computations? 
#test lambda
#test computations for different parameters 


#TESTING FLOOR FOR Gox
#No observations:
"""
dim=1
N=2
obs=np.array([[0.15]])
obs_times=np.array([.5])
x=np.array([[[0.1],[0.3]],[[0.15],[0.5]],[[0.6],[0.7]]])
l=1
Lamb_par=1
g_par=identity(dim).toarray()

print(Gox(obs,obs_times,x,Lambda,Lamb_par,l,N,dim,g_den,g_par))
#print(np.log(np.sqrt(1/(np.pi*2))*np.exp(-1*(0.15-0.5)**2/2)))

"""

#%%
"""
def G(z,x,ht,H,l,para=True):
    #This function emulates the Radon-Nykodim derivative of the Girsanov formula,
    # this formula applies only to the regular-time observations. 
    #ARGUMENTS: z are the observations(2 rank array) , x is the array of the 
    #particles (rank 3 array, with 1 dimension less in the time discretization), 
    #ht is the function that computes the h(x) and d is the distance in which we
    #compute the paths.
    #OUTPUT: logarithm of the weights    
    h=ht(x,H,para=para)
    delta_z=z[1:]-z[:-1]
    delta_l=1./2**l
    suma1=np.einsum("tnd,td->n",h,delta_z)

    suma2=-(1/2.)*delta_l*np.einsum("tnj,tnj->n",h,h)
    #print(suma1,suma2)
    log_w=suma1+suma2
   
    return log_w
"""
#%%
#tests for G
"""
z=np.array([[1,0],[2,0],[3,0]])
H=np.array([[1,0],[0,1]])
x=np.array([[[1,0],[3,0]],[[2,0],[4,0]]])
l=1
#print(ht(x,H))
print(G(z,x,ht,H,l,para=True))

print(x.shape)
times=np.array(range(int(2**l*d+1)))/2**l
plt.plot(times,x0[:,7,0])
#plt.plot(times,x1[:,0,0])


#zg=z[:2**l*d+1]
zg=np.zeros((T*2**l+1,10))+1
#print((T*2**l+1,10))
#print(x.shape)
xg=np.zeros((T*2**l,10,10))
#xg=np.zeros((2560, 10, 10))
lik=G(zg,x[:-1],ht,H,l,para=True)

print(lik)
#
"""

#tests for Gox in one dimension
"""
z=np.array([[1],[2],[3]])
H=np.array([[1]])
x=np.array([[[1],[3]],[[2],[8]],[[5],[6]]])
l=1

obs=np.array([[0],[1],[3]])
obs_times=np.array([.1,.2,.3])

Lamb_par=3
l=1
N=2
dim=1
g_par=np.array([[1.]])
#print(ht(x,H))
Go_x=Gox(obs,obs_times,x,Lambda,Lamb_par,l,N,dim,g_den,g_par)
print(Go_x)


print(x.shape)
times=np.array(range(int(2**l*d+1)))/2**l
plt.plot(times,x0[:,7,0])
#plt.plot(times,x1[:,0,0])


#zg=z[:2**l*d+1]
zg=np.zeros((T*2**l+1,10))+1
#print((T*2**l+1,10))
#print(x.shape)
xg=np.zeros((T*2**l,10,10))
#xg=np.zeros((2560, 10, 10))
lik=G(zg,x[:-1],ht,H,l,para=True)

print(lik)

#"""

#%%

def sr(W,N,x,dim):
    # This function does 2 things, given probability weights (normalized)
    # it uses systematic resampling, and constructs the set of resampled 
    # particles.
    # ARGUMENTS: W: Normalized weights with dimension N (number of particles)
    # x: rank 2 array of the positions of the N particles, its dimesion is
    # Nxdim, where dim is the dimension of the problem.
    # OUTPUTs: part: is a N dimentional array where its value in the ith position
    # represents the number of times that particle was sampled.
    # x_new: is the new set of resampled particles.
    
    Wsum=np.cumsum(W)
    #np.random.seed()
    U=np.random.uniform(0,1./N)
    part=np.zeros(N,dtype=int)
    part[0]=np.floor(N*(Wsum[0]-U)+1)
    k=part[0]
    #print(U,part[0])
    for i in range(1,N):
        j=np.floor(N*(Wsum[i]-U)+1)
        part[i]=j-k
        k=j
    x_new=np.zeros((N,dim))
    k=0
    for i in range(N):
        for j in range(part[i]):
            x_new[k]=x[i]
            k+=1
    return [part, x_new]



def multi_samp_os(W,N,x,dim): #from multinomial sampling order statistics
    # This function does 2 things, given probability weights (normalized)
    # it uses multinomial resampling, and constructs the set of resampled
    # particles. What is the complexity of this function?

    W_cum=np.cumsum(W)
    exps=np.random.exponential(1,N+1)
    exps_cum=np.cumsum(exps)
    os=exps_cum/exps_cum[-1]
    j=0
    part_samp=np.zeros(N,dtype=int)
    #print("os is",os   )
    #print("W_cum is",W_cum)
    for i in range(N):
        w=W_cum[i]
        while w>os[j]:
            part_samp[j]=i
            j+=1
            if j==N:
                print("w is",w)
                print("os is",os[j])
    x_resamp=x[part_samp]

    return [part_samp,x_resamp]


# Test for multi_samp_os

"""
W=np.array([0.1,0.2,0.3,0.4])
N=4
x=np.array([[0,2],[1,4],[2,6],[3,8]])+1
x=x[:,0]
dim=1
print(multi_samp_os(W,N,x,dim))

B=10000
samples=np.array([])
for b in range(B):
    samples=np.concatenate((samples,multi_samp_os(W,N,x,dim)[1]))

plt.hist(samples,bins=np.array([0,1,2,3,4])+1,density=True)
"""


#%%
#%%


def multi_samp(W,N,x,dim): #from multinomial sampling
    # This function does 2 things, given probability weights (normalized)
    # it uses multinomial resampling, and constructs the set of resampled 
    # particles. What is the complexity of this function?
    # In https://stackoverflow.com/questions/40143157/big-o-complexity-of-random-choicelist-in-python3
    # we find that it's O(N log N). We need a different function to find O(N). What is it?
    # ARGUMENTS: W: Normalized weights with dimension N (number of particles)
    # x: rank 2 array of the positions of the N particles, its dimesion is
    # Nxdim, where dim is the dimension of the problem.
    # OUTPUTs: part: is a N dimentional array where its value in the ith position
    # represents the number of times that particle was sampled.
    # x_new: is the new set of resampled particles.
    
    part_samp=np.random.choice(N,size=N,p=W) #particles resampled 
    #print(part_samp)
    x_resamp=x[part_samp]
    return [part_samp,x_resamp] 


"""
#This function is not finished, it is supposed to store the original paths 
#and the resampled paths.
def sr(W,N,x_or,dim,d_steps):
    # This function does 2 things, given probability weights (normalized)
    # it uses systematic resampling, and constructs the set of resampled 
    # particles.
    # ARGUMENTS: W: Normalized weights with dimension N (number of particles)
    # x_or(from x original): rank 3 array of the positions of the N 
    # particles in the discretized interval from i*d_steps:(i+1)*d_steps,
    # its dimesion is Nxdim, where dim is the dimension of the problem.
    # OUTPUTs: part: is a N dimentional array where its value in the ith position
    # represents the number of times that particle was sampled.
    # x_new: is the new set of resampled particles.
    
    Wsum=np.cumsum(W)
    #np.random.seed()
    U=np.random.uniform(0,1./N)
    part=np.zeros(N,dtype=int)
    part[0]=np.floor(N*(Wsum[0]-U)+1)
    k=part[0]
    #print(U,part[0])
    for i in range(1,N):
        j=np.floor(N*(Wsum[i]-U)+1)
        part[i]=j-k
        k=j
    x_new=np.zeros((d_steps,N,dim))
    k=0
    for i in range(N):
        for j in range(part[i]):
            x_new[:,k]=x_or[:,i]
            k+=1
    return [part, x_new]
"""
        
def norm_logweights(lw,ax=0):
    # returns the normalized weights given the log of the normalized weights 
    #ARGUMENTS: lw is a rank 1 array of the weights 
    #OUTPUT: w a rank 1 array of the same dimesion of lw 
    if np.all(np.isnan(lw)):
        raise ValueError("All log-weights are NaN. Cannot normalize.")
    lw = np.nan_to_num(lw, nan=-np.inf)

    m=np.max(lw,axis=ax,keepdims=True)
    wsum=np.sum(np.exp(lw-m),axis=ax,keepdims=True)
    w=np.exp(lw-m)/wsum
    return w

#%%
"""
N=10
W=np.array(range(N))
W=W/np.sum(W)
seed_val=3
x=np.random.multivariate_normal(np.zeros(dim),I,N)
print(W,x)
print(sr(W,N,seed_val,x,dim))
"""

#%%



def PF(T,z,lmax,x0,b_ou,A,Sig_ou,fi,ht,H,l,d,N,dim,resamp_coef,para=True):
    #WARNING!!
    # This function doesn't consider the change in prob when we do not resampsle.
    # Meaning that for the function to work properly we need to resample every single 
    # time, i.e. resamp_coef=1.
    # The particle filter function is inspired in 
    # Bain, A., Crisan, D.: Fundamentals of Stochastic Filtering. Springer,
    # New York (2009).
    # ARGUMENTS: T: final time of the propagation, T>0 and preferably integer
    # z: observation process, its a rank two array with discretized observation
    # at the intervals 2^(-lmax)i, i \in {0,1,...,T2^lmax}. with dimension
    # (T2^{lmax}+1) X dim
    # lmax: level of discretization of the observations
    # x0: initial condition of the particle filter, rank 1 array of dimension
    # dim
    # b_out: function that represents the drift of the process (its specifications
    # is already in the document. A is the arguments taht takes
    # Sig_out: function that represents the diffusion of the process (its specifications
    # is already in the document. Its arguments are included in fi.
    # ht: function in the observation process (its specifications
    # is already in the document). Its arguments are included in H.
    # d: time span in which the resampling is computed. d must be a divisor of T.
    # N: number of particles, N \in naturals greater than 1
    # dim: dimension of the problem
    # para: key to wheter compute the paralelization or not.
    # OUTPUTS: x: is the rank 3 array with the resampled particles at times 
    # 2^{-l}*i, i \in {0,1,..., T*2^l}, its dimentions are (2**l*T+1,N,dim)
    # log_weights: logarithm of the weights at times i*d, for i \in {0,1,...,T/d}.
    # it is a rank 2 array with dimensions (int(T/d),N)
    # suma: its the computation of the particle filter for each dimension of the problem
    # its a rank 2 array with dimensions (int(T/d),dim)
    x=np.zeros((2**l*T+1,N,dim))
    z=cut(T,lmax,l,z)
    log_weights=np.zeros((int(T/d),N))
    x_pf=np.zeros((int(T/d),N,dim))                            
    x_new=x0
    x[0]=x0
    d_steps=int(d*2**l)
    for i in range(int(T/d)):
        x[i*d_steps:(i+1)*d_steps+1]=M(x_new,b_ou,A,Sig_ou,fi,l,d,N,dim)
        xi=x[i*d_steps:(i+1)*d_steps]
        zi=z[i*d_steps:(i+1)*d_steps+1]
        log_weights[i]=G(zi,xi,ht,H,l,para=True)
        weights=norm_logweights(log_weights[i],ax=0)
        #seed_val=i
        #print(weights.shape)
        x_pf[i]=xi[-1]
        ESS=1/np.sum(weights**2)
        if ESS<resamp_coef*N:
            #[part0,part1,x0_new,x1_new]=max_coup_sr(w0,w1,N,xi0[-1],xi1[-1],dim)
            x_new=multi_samp(weights,N,x_pf[i],dim)[1]
        else:
            x_new=x_pf[i]
       #x_new=sr(weights,N,x_pf[i],dim)[1]
    #Filter
    #spots=np.arange(d_steps,2**l*T+1,d_steps,dtype=int)
    #x_pf=x[spots]
    #weights=norm_logweights(log_weights,ax=1)
    #print(x_pf.shape,weights.shape)
    #suma=np.sum(x_pf[:,:,1]*weights,axis=1)
    
    return [x,log_weights,x_pf]


def Cox_PF(T,xin,b_ou,A,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,g_par,Lambda,Lamb_par):
    
    # (T,z,lmax,x0,b_ou,A,Sig_ou,fi,ht,H,l,d,N,dim,resamp_coef,para=True):
    # The particle filter function is inspired in 
    # Bain, A., Crisan, D.: Fundamentals of Stochastic Filtering. Springer,
    # New York (2009).
    # ARGUMENTS: T: final time of the propagation, T>0 and preferably integer
    # z: observation process, its a rank two array with discretized observation
    # at the intervals 2^(-lmax)i, i \in {0,1,...,T2^lmax}. with dimension
    # (T2^{lmax}+1) X dim
    # lmax: level of discretization of the observations
    # x0: initial condition of the particle filter, rank 1 array of dimension
    # dim
    # b_out: function that represents the drift of the process (its specifications
    # is already in the document. A is the arguments taht takes
    # Sig_out: function that represents the diffusion of the process (its specifications
    # is already in the document. Its arguments are included in fi.
    # ht: function in the observation process (its specifications
    # is already in the document). Its arguments are included in H.
    # d: time span in which the resampling is computed. d must be a divisor of T.
    # N: number of particles, N \in naturals greater than 1
    # dim: dimension of the problem
    # para: key to wheter compute the paralelization or not.
    # OUTPUTS: x: is the rank 3 array with the resampled particles at times 
    # 2^{-l}*i, i \in {0,1,..., T*2^l}, its dimentions are (2**l*T+1,N,dim)
    # log_weights: logarithm of the weights at times i*d, for i \in {0,1,...,T/d}.
    # it is a rank 2 array with dimensions (int(T/d),N)
    # suma: its the computation of the particle filter for each dimension of the problem
    # its a rank 2 array with dimensions (int(T/d),dim)
    #x=np.zeros((2**l*T+1,N,dim))
    
    log_weights=np.zeros((int(T/d),N))
    x_pf=np.zeros((int(T/d),N,dim))                            
    x_new=xin
    #x[0]=xin
    #d_steps=int(d*2**l)
    c_indices=np.digitize(obs_time,d*np.array(range(int(T/d)+1)),right=True)-1
    
    for i in range(int(T/d)):
        
        obti=obs_time[np.nonzero(c_indices==i)]
        yi=obs[np.nonzero(c_indices==i)]
        
        #x[i*d_steps:(i+1)*d_steps+1]=M(x_new,b_ou,A,Sig_ou,fi,l,d,N,dim)
        #xi=x[i*d_steps:(i+1)*d_steps+1]
        xi=M(x_new,b_ou,A,Sig_ou,fi,l,d,N,dim)
        #print(xi.shape)
        #print(yi,obti-i*d,)
        #print(x_new,xi)
        #print(xi)
        #Things that could be wrong
        # observations, x_new, weights
        #observations seem to be fine

        log_weights[i]=log_weights[i]+Gox(yi,obti-i*d,xi,Lambda,Lamb_par,l,N,dim,g_den,g_par)
        weights=norm_logweights(log_weights[i],ax=0)
        #print(yi,weights)
        #seed_val=i
        #print(weights.shape)
        x_last=xi[-1]
        x_pf[i]=xi[-1]
        ESS=1/np.sum(weights**2)
        #print(ESS,resamp_coef*N)
        if ESS<resamp_coef*N:
        #if True==False:
            #[part0,part1,x0_new,x1_new]=max_coup_sr(w0,w1,N,xi0[-1],xi1[-1],dim)
            #print(x_new.shape)
            x_new=multi_samp(weights,N,x_last,dim)[1]
            #print(x_new.shape)
        else:
            
            #print("time is",i)
            x_new=x_last
            if i< int(T/d)-1:
                log_weights[i+1]=log_weights[i]
        #print(i)
        
       #x_new=sr(weights,N,x_pf[i],dim)[1]
    #weights=np.reshape(norm_logweights(log_weights,ax=1),(int(T/d),N,1))
    #pf=np.sum(weights*x_pf,axis=1)
    #Filter
    #spots=np.arange(d_steps,2**l*T+1,d_steps,dtype=int)
    #x_pf=x[spots]
    #weights=norm_logweights(log_weights,ax=1)
    #print(x_pf.shape,weights.shape)
    #suma=np.sum(x_pf[:,:,1]*weights,axis=1)
    return [log_weights,x_pf]


def Cox_SM(T,xin,b_ou,A,Sig_ou,fi,Grad_b,b_numb_par,obs,obs_time,l,d,N,dim,resamp_coef,g_den,Grad_Log_g_den,g_par,Lambda,Grad_Lambda,Grad_Log_Lambda,\
    Lamb_par,Lamb_par_numb):

    # (T,z,lmax,x0,b_ou,A,Sig_ou,fi,ht,H,l,d,N,dim,resamp_coef,para=True):
    # The particle filter function is inspired in 
    # Bain, A., Crisan, D.: Fundamentals of Stochastic Filtering. Springer,
    # New York (2009).
    # ARGUMENTS: T: final time of the propagation, T>0 and preferably integer
    # z: observation process, its a rank two array with discretized observation
    # at the intervals 2^(-lmax)i, i \in {0,1,...,T2^lmax}. with dimension
    # (T2^{lmax}+1) X dim
    # lmax: level of discretization of the observations
    # x0: initial condition of the particle filter, rank 1 array of dimension
    # dim
    # b_ou: function that represents the drift of the process (its specifications
    # is already in the document. A is the arguments taht takes
    # Sig_out: function that represents the diffusion of the process (its specifications
    # is already in the document. Its arguments are included in fi.
    # ht: function in the observation process (its specifications
    # is already in the document). Its arguments are included in H.
    # d: time span in which the resampling is computed. d must be a divisor of T.
    # N: number of particles, N \in naturals greater than 1
    # dim: dimension of the problem
    # para: key to wheter compute the paralelization or not.
    # OUTPUTS: x: is the rank 3 array with the resampled particles at times 
    # 2^{-l}*i, i \in {0,1,..., T*2^l}, its dimentions are (2**l*T+1,N,dim)
    # log_weights: logarithm of the weights at times i*d, for i \in {0,1,...,T/d}.
    # it is a rank 2 array with dimensions (int(T/d),N)
    # suma: its the computation of the particle filter for each dimension of the problem
    # its a rank 2 array with dimensions (int(T/d),dim)
    #x=np.zeros((2**l*T+1,N,dim))
    
    log_weights=np.zeros((int(T/d),N)) # log of the particle filter weights at d-spaced 
    # times
    x_pf=np.zeros((int(T/d),N,dim))  # Particles(not resampled) at d-spaced times                        

    x_new=xin # x_new is the initial condition for each iteration, resampled particles
    #x[0]=xin
    d_steps=int(d*2**l)
    c_indices=np.digitize(obs_time,d*np.array(range(int(T/d)+1)),right=True)-1
    # recover the indices for the partinent observation in a certain interval 
    # partitioned by d.

    # In the following three lines we define the variables that will store the gradient,
    # these structures might change and overlap depending on the inclusion of the parameters 
    # in the b, lambda, and g_den function.


    F_Lambda=np.zeros((int(T/d),N,Lamb_par_numb)) #Function F corresponding to the parameters 
    # of Lambda, the intensity funciton of the Cox process
    F_g_den=np.zeros((int(T/d),N,dim,dim)) # Function F corresponding to the parameters 
    # of observation likelihood function g(y|x;g_par)
    F_b=np.zeros((int(T/d),N,b_numb_par)) # Function F correspoding to the parameters of the 
    # drift, in this case, A. 
    f_Lambda=F_Lambda[0]
    f_g_den=F_g_den[0]
    f_b=F_b[0]
    samp_par=np.array(range(N))
    for i in range(int(T/d)):
        # in the following two lines we get the observations and observation times for the interval
        # (i*d,(i+1)*d)
        obti=obs_time[np.nonzero(c_indices==i)]
        yi=obs[np.nonzero(c_indices==i)]
        # we propagate the particles and obtain the term2 function (the term of F that involves the brownian motions)
        xi,term2A,term2B=M_smooth(x_new,b_ou,A,Sig_ou,fi,Grad_b,b_numb_par,l,d,N,dim)
         
        term1A=-np.sum(Grad_Lambda(xi[:1],Lamb_par,Lamb_par_numb,ax1=-1),axis=0)/2**l
        term1B=-np.sum(Grad_Lambda(xi[1:-1],Lamb_par,Lamb_par_numb,ax1=-1),axis=0)/2**l
        
        Log_trans=log_trans(xi[0],xi[1],b_ou,A,Sig_ou,fi,l)
        #print("log_trans is",Log_trans)
        log_w ,log_w_init, term3_Lambd,term3_g_den=Gox_SM(yi,obti-i*d,xi,Lambda\
        ,Grad_Log_Lambda,Lamb_par,Lamb_par_numb,l,N,dim,g_den,Grad_Log_g_den,g_par)
        log_weights[i]=log_weights[i]+log_w
        #print("log_w_init is ",log_w_init)
        weights=norm_logweights(log_weights[i],ax=0)
        if log_w_init.ndim==1:
            #Lambda parameters 
            #mini_w=norm_logweights(Log_trans+log_w_init[:,np.newaxis],ax=0)
            mini_w=norm_logweights(Log_trans+log_w_init[:,np.newaxis],ax=0)
            #print("mini_w is ",mini_w)
            f_LambdaA=np.einsum("ji,jp->ip",mini_w,f_Lambda[samp_par]+term1A)
            f_LambdaB=term1B +term3_Lambd
            f_Lambda= f_LambdaA+f_LambdaB
            
            F_Lambda[i]=f_Lambda
            #print("f_Lambda is ",f_Lambda)
            f_bA=np.einsum("ji,jp->ip",mini_w,f_b[samp_par]+term2A)
            f_bB=term2B
            f_b=f_bA+f_bB
            F_b[i]=f_b
            print(f_b)
            #print("f_b is ",f_b)
            f_g_denA=np.einsum("ji,jpq->ipq",mini_w,f_g_den[samp_par])
            f_g_denB=term3_g_den
            f_g_den=f_g_denA+f_g_denB
            F_g_den[i]=f_g_den
            #print("f_g_den is ",f_g_den)

        if log_w_init.ndim==2:
            #print("log_w_init",log_w_init)
            mini_w=norm_logweights(Log_trans+log_w_init,ax=0)
            #print("mini_w is ",mini_w)
            f_Lambda=np.einsum("ji,jip->ip",mini_w,\
            (f_Lambda[samp_par]+term1A)[:,np.newaxis]+term1B+term3_Lambd)
            #print( "vec is",(term1A)[:,np.newaxis]+term1B)
            
            F_Lambda[i]=f_Lambda
            #print("f_Lambda is ",f_Lambda)
            f_bA=np.einsum("ji,jp->ip",mini_w,f_b[samp_par]+term2A)
            f_bB=term2B
            f_b=f_bA+f_bB
            F_b[i]=f_b
            #print("f_b is ",f_b)
            f_g_den=np.einsum("ji,jipq->ipq",mini_w,\
            (f_g_den[samp_par])[:,np.newaxis]+term3_g_den)
            #print("term3_g_den is ",term3_g_den)
            #print("f_g_den is ",f_g_den)
            F_g_den[i]=f_g_den




        x_last=xi[-1]
        x_pf[i]=xi[-1]
        ESS=1/np.sum(weights**2)
        if ESS<resamp_coef*N:
            [samp_par,x_new]=multi_samp(weights,N,x_last,dim)
        else:
            x_new=x_last
            if i< int(T/d)-1:
                log_weights[i+1]=log_weights[i]
            samp_par=np.array(range(N))
    return [log_weights,x_pf,F_Lambda,F_b,F_g_den]

def Cox_SM_W(T,xin,b_ou,A,Sig_ou,fi,Grad_b,b_numb_par,obs,obs_time,l,d,N,dim,resamp_coef,g_den,Grad_Log_g_den,g_par,Lambda,Grad_Lambda,Grad_Log_Lambda,\
    Lamb_par,Lamb_par_numb):
    #This version is different from Cox_SM in the computation of the smoother,
    # where here instead of computing the resampled particles we multiply the weights 
    #Memory friendly version where instead of 3 ouputs we just output the pf.
    # (T,z,lmax,x0,b_ou,A,Sig_ou,fi,ht,H,l,d,N,dim,resamp_coef,para=True):
    # The particle filter function is inspired in 
    # Bain, A., Crisan, D.: Fundamentals of Stochastic Filtering. Springer,
    # New York (2009).
    # ARGUMENTS: T: final time of the propagation, T>0 and preferably integer
    # z: observation process, its a rank two array with discretized observation
    # at the intervals 2^(-lmax)i, i \in {0,1,...,T2^lmax}. with dimension
    # (T2^{lmax}+1) X dim
    # lmax: level of discretization of the observations
    # x0: initial condition of the particle filter, rank 1 array of dimension
    # dim
    # b_out: function that represents the drift of the process (its specifications
    # is already in the document. A is the arguments taht takes
    # Sig_out: function that represents the diffusion of the process (its specifications
    # is already in the document. Its arguments are included in fi.
    # ht: function in the observation process (its specifications
    # is already in the document). Its arguments are included in H.
    # d: time span in which the resampling is computed. d must be a divisor of T.
    # N: number of particles, N \in naturals greater than 1
    # dim: dimension of the problem
    # para: key to wheter compute the paralelization or not.
    # OUTPUTS: x: is the rank 3 array with the resampled particles at times 
    # 2^{-l}*i, i \in {0,1,..., T*2^l}, its dimentions are (2**l*T+1,N,dim)
    # log_weights: logarithm of the weights at times i*d, for i \in {0,1,...,T/d}.
    # it is a rank 2 array with dimensions (int(T/d),N)
    # suma: its the computation of the particle filter for each dimension of the problem
    # its a rank 2 array with dimensions (int(T/d),dim)
    #x=np.zeros((2**l*T+1,N,dim))
    
    log_weights=np.zeros((int(T/d),N)) # log of the particle filter weights at d-spaced 
    # times
    x_pf=np.zeros((int(T/d),N,dim))  # Particles(not resampled) at d-spaced times                        

    x_new=xin
    x_last=np.zeros((N,dim))+xin # x_new is the initial condition for each iteration, resampled particles
    #x[0]=xin
    d_steps=int(d*2**l)
    c_indices=np.digitize(obs_time,d*np.array(range(int(T/d)+1)),right=True)-1
    # recover the indices for the partinent observation in a certain interval 
    # partitioned by d.

    # In the following three lines we define the variables that will store the gradient,
    # these structures might change and overlap depending on the inclusion of the parameters 
    # in the b, lambda, and g_den function.


    F_Lambda=np.zeros((int(T/d),N,Lamb_par_numb)) #Function F corresponding to the parameters 
    # of Lambda, the intensity funciton of the Cox process
    F_g_den=np.zeros((int(T/d),N,dim,dim)) # Function F corresponding to the parameters 
    # of observation likelihood function g(y|x;g_par)
    F_b=np.zeros((int(T/d),N,b_numb_par)) # Function F correspoding to the parameters of the 
    # drift, in this case, A. 
    f_Lambda=F_Lambda[0]
    f_g_den=F_g_den[0]
    f_b=F_b[0]
    samp_par=np.array(range(N))
    log_w_prev=log_weights[0]
    for i in range(int(T/d)):
        # in the following two lines we get the observations and observation times for the interval
        # (i*d,(i+1)*d)
        obti=obs_time[np.nonzero(c_indices==i)]
        yi=obs[np.nonzero(c_indices==i)]
        # we propagate the particles and obtain the term2 function (the term of F that involves the brownian motions)
        xi,term2A,term2B=M_smooth_W(x_new,x_last,b_ou,A,Sig_ou,fi,Grad_b,b_numb_par,l,d,N,dim)
        #print("term2A is ",term2A)
        #print("term2B is ",term2B)
        term1A=-np.sum(Grad_Lambda(x_last[np.newaxis,:,:],Lamb_par,Lamb_par_numb,ax1=-1),axis=0)/2**l
        term1B=-np.sum(Grad_Lambda(xi[1:-1],Lamb_par,Lamb_par_numb,ax1=-1),axis=0)/2**l
        Log_trans=log_trans(x_last,xi[1],b_ou,A,Sig_ou,fi,l)
        #print(Log_trans.shape)
        #print("log_trans is",Log_trans)
        log_w ,log_w_init, term3_Lambd,term3_g_den=Gox_SM_W(yi,obti-i*d,xi,x_last,samp_par, Lambda\
        ,Grad_Log_Lambda,Lamb_par,Lamb_par_numb,l,N,dim,g_den,Grad_Log_g_den,g_par)
        

        

        if log_w_init.ndim==1:
            #Lambda parameters 
            #mini_w=norm_logweights(Log_trans+log_w_init[:,np.newaxis],ax=0)
            #print(log_w_prev[:,np.newaxis].shape)
            mini_w=norm_logweights(Log_trans+log_w_init[:,np.newaxis]+log_w_prev[:,np.newaxis],ax=0)
            #print("mini_w is ",mini_w)
            f_LambdaA=np.einsum("ji,jp->ip",mini_w,f_Lambda+term1A)
            f_LambdaB=term1B +term3_Lambd
            f_Lambda= f_LambdaA+f_LambdaB
            
            F_Lambda[i]=f_Lambda
            #print("f_Lambda is ",f_Lambda)
            f_bA=np.einsum("ji,jp->ip",mini_w,f_b+term2A)
            f_bB=term2B
            f_b=f_bA+f_bB
            print("f_bB is ",f_bB)
            #print("f_b shape is ",f_b.shape)
            F_b[i]=f_b


            f_g_denA=np.einsum("ji,jpq->ipq",mini_w,f_g_den)
            f_g_denB=term3_g_den
            f_g_den=f_g_denA+f_g_denB
            F_g_den[i]=f_g_den
            #print("f_g_den is ",f_g_den)

        if log_w_init.ndim==2:
            #print("log_w_init",log_w_init)
            mini_w=norm_logweights(Log_trans+log_w_init+log_w_prev[:,np.newaxis],ax=0)
            #print("mini_w is ",mini_w)
            f_Lambda=np.einsum("ji,jip->ip",mini_w,\
            (f_Lambda+term1A)[:,np.newaxis]+term1B+term3_Lambd)
            #print( "vec is",(term1A)[:,np.newaxis]+term1B)
            
            F_Lambda[i]=f_Lambda
            #print("f_Lambda is ",f_Lambda)
            f_bA=np.einsum("ji,jp->ip",mini_w,f_b+term2A)
            f_bB=term2B
            f_b=f_bA+f_bB
            F_b[i]=f_b
            #print("f_b is ",f_b)
            f_g_den=np.einsum("ji,jipq->ipq",mini_w,\
            (f_g_den[samp_par])+term3_g_den)
            #print("term3_g_den is ",term3_g_den)
            #print("f_g_den is ",f_g_den)
            F_g_den[i]=f_g_den



        log_weights[i]=log_weights[i]+log_w
        #print("log_w_init is ",log_w_init)
        weights=norm_logweights(log_weights[i],ax=0)
        x_last=xi[-1]
        x_pf[i]=xi[-1]
        ESS=1/np.sum(weights**2)
        if ESS<resamp_coef*N:
            [samp_par,x_new]=multi_samp(weights,N,x_last,dim)
        else:
            x_new=x_last
            if i< int(T/d)-1:
                log_weights[i+1]=log_weights[i]
            samp_par=np.array(range(N))

        log_w_prev=log_weights[i]
    return [log_weights,x_pf,F_Lambda,F_b,F_g_den]
#%%
# test w
"""
np.random.seed(5)
T=4000
dim=1
dim_o=dim
xin=np.zeros(dim)+1
l=6
collection_input=[]
I=identity(dim).toarray()
df=1
fi=np.array([[1]])
collection_input=[dim, b_lan,df,Sig_ou,fi]
cov=I*1e0
g_pars=[dim,cov]
g_par=cov
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=5
[obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_pars)
print(obs_time,obs,len(obs_time))
times=2**(-l)*np.array(range(int(T*2**l+1)))
b_numb_par=1
Lamb_par_numb=1
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
plt.legend()
d=2**(0)
N=40
"""
# %%

#%%

"""df_init=df+1
Lamb_par_init=Lamb_par+2
g_par_init=g_par+3
print(df_init,Lamb_par_init,g_par_init)
d=1
N=4
results=Cox_SM_W(T,xin,b_lan,df,Sig_ou,fi,Grad_b_lan,b_numb_par,obs,obs_time,l,d,N,dim,resamp_coef,\
g_den,Grad_Log_g_den,g_par,Lambda,Grad_Lambda,Grad_Log_Lambda,Lamb_par,Lamb_par_numb)

#print(results[3][-1])
print(np.sum(results[2][-1]*(norm_logweights(results[0][-1],ax=0)[:,np.newaxis]),axis=0)/T)
print(np.sum(results[3][-1]*(norm_logweights(results[0][-1],ax=0)[:,np.newaxis]),axis=0)/T)
print(np.sum(results[4][-1]*(norm_logweights(results[0][-1],ax=0)[:,np.newaxis,np.newaxis]),axis=0)/T)
df_init=df+1.5
Lamb_par_init=Lamb_par+2
g_par_init=g_par+3
b_numb_par=1
Lamb_par_numb=1
N=1000
l=6 
d=2
resamp_coef=0.8
beta=2.9/4.
step_0=1
b_step=30
results=Cox_PE(T,xin,b_lan,df_init,Sig_ou,fi,Grad_b_lan,b_numb_par,obs,obs_time,l,d,N,\
dim,resamp_coef,g_den,Grad_Log_g_den,g_par_init,Lambda,Grad_Lambda,Grad_Log_Lambda,\
Lamb_par_init,Lamb_par_numb,step_0,beta)

"""
#%%
"""
Lambda_pars,g_pars,dfs=results[2],results[3],results[4]
print(Lamb_par,Lamb_par_init)
print("g_par",g_par[0,0])
print(Lambda_pars)
plt.plot(range(int(T/d)+1),Lambda_pars,label=r"$\theta_\lambda$")
plt.plot(range(int(T/d)+1),Lamb_par+np.zeros(int(T/d)+1),label=r"$\bar{\theta}_\lambda$",ls="--")
plt.xlabel("Iterations")
#plt.ylabel(r"$\theta_\lambda$")
print(g_par,g_par_init)
print(g_pars)
plt.plot(range(int(T/d)+1),g_pars[:,0,0],label=r"$\theta_\Sigma$")
plt.plot(range(int(T/d)+1),g_par[0,0]+np.zeros(int(T/d)+1),label=r"$\bar{\theta}_\Sigma$",ls="--")
print(df,df_init)
print(dfs)
plt.plot(range(int(T/d)+1),dfs,label=r"$\theta_b$")
plt.plot(range(int(T/d)+1),df+np.zeros(int(T/d)+1),label=r"$\bar{\theta}_b$",ls="--")
plt.legend()
#plt.savefig("images/SGD_ou_T4000_d2_N1000_l8.pdf")

"""
#%%
"""
np.random.seed(2)
T=4000
dim=1
dim_o=dim
xin=np.zeros(dim)+0.1
l=8
collection_input=[]
I=identity(dim).toarray()
c=2**(-6)
mu=np.abs(np.random.normal(1.01,0,dim))
sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
mu=mu*c
sigs=sigs*np.sqrt(c)
print(mu,sigs)
comp_matrix=np.array([[1]])
inv_mat=comp_matrix.T
S=diags(np.random.normal(1,0,dim),0).toarray()
Sig=inv_mat@S@comp_matrix
fi=[sigs,Sig]
collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
cov=I*1e0
g_pars=[dim,cov]
g_par=cov
samples=5
np.random.seed(5)
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=.5
[obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_pars)
print(len(obs))
times=2**(-l)*np.array(range(int(T*2**l+1)))
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
plt.legend()
plt.show()
"""
#%%
"""
np.random.seed(2)
T=4000
dim=1
dim_o=dim
xin=np.zeros(dim)+0.1
l=8
collection_input=[]
I=identity(dim).toarray()
c=2**(-6)
mu=np.abs(np.random.normal(1.01,0,dim))
sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
mu=mu*c
sigs=sigs*np.sqrt(c)
print(mu,sigs)
comp_matrix=np.array([[1]])
inv_mat=comp_matrix.T
S=diags(np.random.normal(1,0,dim),0).toarray()
Sig=inv_mat@S@comp_matrix
fi=[sigs,Sig]
collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
cov=I*1e0
g_pars=[dim,cov]
g_par=cov
samples=5
np.random.seed(5)
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=.5
[obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_pars)
print(len(obs))
times=2**(-l)*np.array(range(int(T*2**l+1)))
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
plt.legend()
plt.show()
"""
#%%
"""
print(mu)
print(Lamb_par)
print(g_par)
"""
#%%
#%%
"""
np.random.seed(1)
mu_init=mu+1
Lamb_par_init=Lamb_par+2
g_par_init=g_par+1
b_numb_par=1
Lamb_par_numb=1
N=1000
l=8
d=1
resamp_coef=0.8
beta=2.4/4.
step_0=.1
Lamb_step=1
b_step=1
g_step=12
results=Cox_PE(T,xin,b_gbm,mu_init,Sig_gbm,fi,Grad_b_gbm,b_numb_par,obs,obs_time,l,d,N,\
dim,resamp_coef,g_den,Grad_Log_g_den,g_par_init,Lambda,Grad_Lambda,Grad_Log_Lambda,\
Lamb_par_init,Lamb_par_numb,step_0,beta)
""" 
#%%
"""
np.savetxt("Observations&data/SGD_gbm_T1000_dp5_N1000_l8_betap6_Lambda_pars_v2.txt",results[2],fmt="%f")
np.savetxt("Observations&data/SGD_gbm_T1000_dp5_N1000_l8_betap6_g_pars_v2.txt",results[3].flatten(),fmt="%f")
np.savetxt("Observations&data/SGD_gbm_T1000_dp5_N1000_l8_betap6_As_v2.txt",results[4],fmt="%f")
"""
#%%
"""
T=4000
d=1
g_pars=np.reshape(np.loadtxt("Observations&data/SGD_gbm_T1000_dp5_N1000_l8_betap6_g_pars_v2.txt"),(int(T/d)+1,dim,dim))
Lambda_pars=np.loadtxt("Observations&data/SGD_gbm_T1000_dp5_N1000_l8_betap6_Lambda_pars_v2.txt")
mus=np.loadtxt("Observations&data/SGD_gbm_T1000_dp5_N1000_l8_betap6_As_v2.txt")
"""
#%%
""" 
lw=3
fontsize=14
left_lim=4000
Lambda_pars,g_pars,mus=results[2],results[3],results[4]
print(Lamb_par,Lamb_par_init)
print("g_par",g_par[0,0])
#print(Lambda_pars)
plt.plot(range(int(T/d)+1)[:left_lim],Lambda_pars[:left_lim],label=r"$\theta_\lambda$",c="coral",lw=lw)
plt.plot(range(int(T/d)+1)[:left_lim],(Lamb_par+np.zeros(int(T/d)+1))[:left_lim],label=r"$\bar{\theta}_\lambda$",ls="--",c="red",lw=lw)
plt.xlabel("Iterations",fontsize=fontsize)
#plt.ylabel(r"$\theta_\lambda$")
print(g_par,g_par_init)
#print(g_pars)
plt.plot(range(int(T/d)+1)[:left_lim],g_pars[:left_lim,0,0],label=r"$\theta_\Sigma$",c="deepskyblue",lw=lw)
plt.plot(range(int(T/d)+1)[:left_lim],(g_par[0,0]+np.zeros(int(T/d)+1))[:left_lim],label=r"$\bar{\theta}_\Sigma$",ls="dashdot",c="blue",lw=lw)
print(mu,mu_init)
#print(mu)
plt.plot(range(int(T/d)+1)[:left_lim],mus[:left_lim],label=r"$\theta_b$",c="mediumturquoise",lw=lw)
plt.plot(range(int(T/d)+1)[:left_lim],(mu[0]+np.zeros(int(T/d)+1))[:left_lim],label=r"$\bar{\theta}_b$",ls="dotted",c="green",lw=lw)
plt.legend(fontsize=fontsize)
plt.title("GBM",fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.locator_params(axis="x",nbins=6)
#plt.show()
plt.savefig("Images/SGD_gbm_T1000_dp5_N1000_l8_betap6_paper_v2.pdf")

"""
#%%
"""
np.random.seed(0)
T=100
dim=1
dim_o=dim
xin=np.zeros(dim)+0.1
l=13
collection_input=[]
I=identity(dim).toarray()
c=2**(-6)
mu=np.abs(np.random.normal(1.01,0,dim))
sigs=np.abs(np.random.normal(np.sqrt(2),0,dim))
mu=mu*c
sigs=sigs*np.sqrt(c)
print(mu,sigs)
comp_matrix=np.array([[1]])
inv_mat=comp_matrix.T
S=diags(np.random.normal(1,0,dim),0).toarray()
Sig=inv_mat@S@comp_matrix
fi=[sigs,Sig]
collection_input=[dim, b_gbm,mu,Sig_gbm,fi]
cov=I*1e0
g_pars=[dim,cov]
g_par=cov
samples=5
np.random.seed(0)
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=.5
[obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_pars)
print(len(obs))
times=2**(-l)*np.array(range(int(T*2**l+1)))
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
plt.legend()
plt.show()
"""
#%%
"""
df_init=df-0.5
Lamb_par_init=Lamb_par+2
g_par_init=g_par+3
b_numb_par=1
Lamb_par_numb=1
N=1000
l=8 
d=2
resamp_coef=0.8
beta=3/4.
step_0=1
b_step=2

results=Cox_PE(T,xin,b_gbm,df_init,Sig_gbm,fi,Grad_b_gbm,b_numb_par,obs,obs_time,l,d,N,\
dim,resamp_coef,g_den,Grad_Log_g_den,g_par_init,Lambda,Grad_Lambda,Grad_Log_Lambda,\
Lamb_par_init,Lamb_par_numb,step_0,beta)   
""" 

#%%
"""
np.random.seed(0)
T=2
N=2
dim=1
xin=np.array([[1],[2]])
A=np.array([[2]])
fi=2*A
obs=np.array([[1],[2],[-1]])
obs_time=np.array([0.25,0.8,1.25])
l=1
d=1
resamp_coef=0.8
g_par=identity(dim).toarray()*2
Lamb_par=2
Lamb_par_numb=1
results=Cox_SM(T,xin,b_ou,A,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,Grad_Log_g_den,g_par,Lambda,Grad_Lambda,Grad_Log_Lambda,\
Lamb_par,Lamb_par_numb)
"""

# RESULTS: THE CODE WAS COMPARED WITH MANUAL COMPUTATIONS, THE VALUES FOR THE PROBABILITIES TERMS G AND M WERE COMPARED,
# THE GRADIENTS WERE COMPARED, AND THE RESULTS ARE CORRECT. THE CODE IS WORKING PROPERLY WITH THIS SETTING.

#%%
# HERE WE COMPARE THE COX_SMOOTHER WITH A SI FILTER, I.E. A FILTER WITHOUT RESAMPLING.

#testo
# First we define the state space model
"""
np.random.seed(5)
T=1
dim=1
dim_o=dim
xin=np.zeros(dim)+1
l=8
resamp_coef=0.8
collection_input=[]
I=identity(dim).toarray()
#comp_matrix = ortho_group.rvs(dim)
comp_matrix=np.array([[1]])
inv_mat=la.inv(comp_matrix)
#S=diags(np.random.normal(1,0.1,dim),0).toarray()
S=diags(np.random.normal(0.99,0.1,dim),0).toarray()
#S=np.array([[1.]])
fi=inv_mat@S@comp_matrix
#B=diags(np.random.normal(-1,0.1,dim),0).toarray()
B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
B=inv_mat@B@comp_matrix
#B=np.array([[0.8]])
B_true=B
B_init=B_true
#B=np.array([[-1.]])
#print(B)
#print(B)
#B=comp_matrix-comp_matrix.T  +B 
collection_input=[dim, b_ou,B,Sig_ou,fi]
cov=I*1e0
g_pars=[dim,cov]
g_par=cov
g_par_init=cov
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=5.5
Lamb_par_init=Lamb_par
[obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_pars)
print(obs_time,obs,len(obs_time))
#plt.plot(times,x_true)
times=2**(-l)*np.array(range(int(T*2**l+1)))
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
#plt.plot(times,kbf,label="KBF")
#plt.plot(times,kf,label="KF")
#plt.plot(times,kf-kbf)
#print(len(obs_time))
plt.legend()
"""
#%%
# In this cell we check that the gradient when the initial parameters are the same as the true parameters
# is a distribution with mean zero, where randomness comes from the observations.

"""
B=1000
Grads_obs_dep_b=np.zeros((B,dim))
Grads_obs_dep_Lambda=np.zeros((B,dim))
Grads_obs_dep_g_den=np.zeros((B,dim))
for i in range(B):
    np.random.seed(i)
    [obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_pars)
    d=T
    N_is=100
    Lamb_par_numb=1
    [x,term2A,term2B]= M_smooth(xin,b_ou,B_init,Sig_ou,fi,l,d,N_is,dim)
    [log_w ,log_w_init, term3_Lambd,term3_g_den]=Gox_SM(obs,obs_time,x,Lambda\
    ,Grad_Log_Lambda,Lamb_par_init,Lamb_par_numb,l,N_is,dim,g_den,Grad_Log_g_den,g_par_init)
    term1=-np.sum(Grad_Lambda(x[:-1],Lamb_par_init,Lamb_par_numb,ax1=-1),axis=0)/2**l
    F_Lambda=np.sum((term1+term3_Lambd)*norm_logweights(log_w,ax=0)[:,np.newaxis],axis=0)

    F_b=np.sum((term2A+term2B)*norm_logweights(log_w,ax=0)[:,np.newaxis,np.newaxis],axis=0)

    F_g_den=np.sum(term3_g_den*norm_logweights(log_w,ax=0)[:,np.newaxis,np.newaxis],axis=0)

    Grads_obs_dep_b[i]=F_b[0]
    Grads_obs_dep_Lambda[i]=F_Lambda[0]
    Grads_obs_dep_g_den[i]=F_g_den[0]
    print(i)

plt.hist(Grads_obs_dep_b, bins=30,density=True) 
print(np.mean(Grads_obs_dep_b,axis=0))
plt.hist(Grads_obs_dep_Lambda, bins=30,density=True) 
print(np.mean(Grads_obs_dep_Lambda,axis=0))
plt.hist(Grads_obs_dep_g_den, bins=30,density=True) 
print(np.mean(Grads_obs_dep_g_den,axis=0))
"""
# RESULTS: The histograms show that the gradients are centered at zero( with small variances of these
# depending on the specific realization of the signal), so the code is working properly.
#%%
"""
N=1000
d=1
Lamb_par_numb=1
np.random.seed(1)
results=Cox_SM_W(T,xin,b_ou,B_init,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,\
Grad_Log_g_den,g_par_init,Lambda,Grad_Lambda,Grad_Log_Lambda,Lamb_par_init,Lamb_par_numb)
"""
#%%
# IN this cell we compute the PF and compare wit hthe PF of the smoother to check that the results are the same.
"""
N=1000
d=1
Lamb_par_numb=1
np.random.seed(5)
results_PF=Cox_PF(T,xin,b_ou,B_init,Sig_ou,fi,obs,obs_time,l,d,N,dim,resamp_coef,g_den,\
        g_par_init,Lambda,Lamb_par_init)
"""
#%%
"""
print("The results from the PF are")
print(np.sum(norm_logweights( results_PF[0][-2])*results_PF[1][-2,:,0]))
print("The results from the PF of the smoother are")
print(np.sum(norm_logweights( results[0][-2])*results[1][-2,:,0]))
"""
# RESULTS: The results from Cox_PF and Cox_SM_W are the same, so the code is working properly, regarding
# its PF. 

#%% IN THIS PART WE GET THE GRADIENTS OF THE SMOOTHING DISTRIBUTION WITH IS
# WHAT DO WE NEED? THE PATHS, THE BROWNIAN MOTION, THE INTERPOLATIONS

"""np.random.seed(5)
d=T
N_is=100
[x,term2A,term2B]= M_smooth(xin,b_ou,B_init,Sig_ou,fi,l,d,N_is,dim)
[log_w ,log_w_init, term3_Lambd,term3_g_den]=Gox_SM(obs,obs_time,x,Lambda\
,Grad_Log_Lambda,Lamb_par_init,Lamb_par_numb,l,N_is,dim,g_den,Grad_Log_g_den,g_par_init)
term1=-np.sum(Grad_Lambda(x[:-1],Lamb_par_init,Lamb_par_numb,ax1=-1),axis=0)/2**l
"""
#%%
"""
print("The results from the IS are")
F_Lambda=np.sum((term1+term3_Lambd)*norm_logweights(log_w,ax=0)[:,np.newaxis],axis=0)
print(F_Lambda/T)
F_b=np.sum((term2A+term2B)*norm_logweights(log_w,ax=0)[:,np.newaxis,np.newaxis],axis=0)
print(F_b/T)
F_g_den=np.sum(term3_g_den*norm_logweights(log_w,ax=0)[:,np.newaxis,np.newaxis],axis=0)
print(F_g_den/T)
print("The results from the SMOOTHER are")
print((norm_logweights(results[0],ax=0)[:,np.newaxis]).shape)

"""
# RESULTS: The results form the IS and the smoother are the same, so the code is working properly.
# RESULTS: The tests show that the sign of the gradient is correct, this was done changing the values of the 
# initial parameters in comparation with the true parameters. 
#%%


# NOW WE CHANGE THE PARAMETERS DEPENDING OF THE PARAMETERS OF THE SMOOTHER


def Cox_PE(T,xin,b_ou,A,Sig_ou,fi,Grad_b,b_numb_par,obs,obs_time,l,d,N,dim,resamp_coef,g_den,Grad_Log_g_den,g_par,Lambda,Grad_Lambda,Grad_Log_Lambda,\
    Lamb_par,Lamb_par_numb,step_0,beta):
    # This version takes Cox_SM_W and introduces the changes in the parameters 
    # so we can adaptatively estimate and improve the estimation
    #Memory friendly version where instead of 3 ouputs we just output the pf.
    # (T,z,lmax,x0,b_ou,A,Sig_ou,fi,ht,H,l,d,N,dim,resamp_coef,para=True):
    # The particle filter function is inspired in 
    # Bain, A., Crisan, D.: Fundamentals of Stochastic Filtering. Springer,
    # New York (2009).
    # ARGUMENTS: T: final time of the propagation, T>0 and preferably integer
    # z: observation process, its a rank two array with discretized observation
    # at the intervals 2^(-lmax)i, i \in {0,1,...,T2^lmax}. with dimension
    # (T2^{lmax}+1) X dim
    # lmax: level of discretization of the observations
    # x0: initial condition of the particle filter, rank 1 array of dimension
    # dim
    # b_out: function that represents the drift of the process (its specifications
    # is already in the document. A is the arguments taht takes
    # Sig_out: function that represents the diffusion of the process (its specifications
    # is already in the document. Its arguments are included in fi.
    # ht: function in the observation process (its specifications
    # is already in the document). Its arguments are included in H.
    # d: time span in which the resampling is computed. d must be a divisor of T.
    # N: number of particles, N \in naturals greater than 1
    # dim: dimension of the problem
    # para: key to wheter compute the paralelization or not.
    # OUTPUTS: x: is the rank 3 array with the resampled particles at times 
    # 2^{-l}*i, i \in {0,1,..., T*2^l}, its dimentions are (2**l*T+1,N,dim)
    # log_weights: logarithm of the weights at times i*d, for i \in {0,1,...,T/d}.
    # it is a rank 2 array with dimensions (int(T/d),N)
    # suma: its the computation of the particle filter for each dimension of the problem
    # its a rank 2 array with dimensions (int(T/d),dim)
    #x=np.zeros((2**l*T+1,N,dim))
    log_weights=np.zeros((int(T/d),N)) # log of the particle filter weights at d-spaced 
    # times
    x_pf=np.zeros((int(T/d),N,dim))  # Particles(not resampled) at d-spaced times                        
    x_new=xin
    x_last=np.zeros((N,dim))+xin # x_new is the initial condition for each iteration, resampled particles
    #x[0]=xin
    d_steps=int(d*2**l)
    c_indices=np.digitize(obs_time,d*np.array(range(int(T/d)+1)),right=True)-1
    # recover the indices for the partinent observation in a certain interval 
    # partitioned by d.
    # Values of the gradient descent parameters
    Lambda_pars=np.zeros((int(T/d)+1,Lamb_par_numb))
    Lambda_pars[0]=Lamb_par
    g_pars=np.zeros((int(T/d)+1,dim,dim))
    g_pars[0]=g_par
    As=np.zeros((int(T/d)+1,b_numb_par))
    As[0]=A
    # In the following three lines we define the variables that will store the gradient,
    # these structures might change and overlap depending on the inclusion of the parameters 
    # in the b, lambda, and g_den function.
    F_Lambda=np.zeros((int(T/d)+1,N,Lamb_par_numb)) #Function F corresponding to the parameters 
    # of Lambda, the intensity funciton of the Cox process
    F_g_den=np.zeros((int(T/d)+1,N,dim,dim)) # Function F corresponding to the parameters 
    # of observation likelihood function g(y|x;g_par)
    F_b=np.zeros((int(T/d)+1,N,b_numb_par)) # Function F correspoding to the parameters of the 
    # drift, in this case, A. 
    f_Lambda=F_Lambda[0]
    f_g_den=F_g_den[0]
    f_b=F_b[0]
    Grads_Lambda=np.zeros((int(T/d)+1,Lamb_par_numb)) 
    Grads_g_den=np.zeros((int(T/d)+1,dim,dim))
    Grads_b=np.zeros((int(T/d)+1,b_numb_par))
    samp_par=np.array(range(N))
    log_w_prev=log_weights[0]
    for i in range(int(T/d)):
        # in the following two lines we get the observations and observation times for the interval
        # (i*d,(i+1)*d)
        A=As[i]
        # Uncomment the following line when computing the OU configuration.
        #A=A[:,np.newaxis]
        g_par=g_pars[i]
        Lamb_par=Lambda_pars[i]
        obti=obs_time[np.nonzero(c_indices==i)]
        yi=obs[np.nonzero(c_indices==i)]
        # we propagate the particles and obtain the term2 function (the term of F that involves the brownian motions)
        xi,term2A,term2B=M_smooth_W(x_new,x_last,b_ou,A,Sig_ou,fi,Grad_b,b_numb_par,l,d,N,dim)
        term1A=-np.sum(Grad_Lambda(x_last[np.newaxis,:,:],Lamb_par,Lamb_par_numb,ax1=-1),axis=0)/2**l
        term1B=-np.sum(Grad_Lambda(xi[1:-1],Lamb_par,Lamb_par_numb,ax1=-1),axis=0)/2**l
        Log_trans=log_trans(x_last,xi[1],b_ou,A,Sig_ou,fi,l)
        #print(Log_trans.shape)
        #print("log_trans is",Log_trans)
        log_w ,log_w_init, term3_Lambd,term3_g_den=Gox_SM_W(yi,obti-i*d,xi,x_last,samp_par, Lambda\
        ,Grad_Log_Lambda,Lamb_par,Lamb_par_numb,l,N,dim,g_den,Grad_Log_g_den,g_par)
        if log_w_init.ndim==1:
            #Lambda parameters 
            #mini_w=norm_logweights(Log_trans+log_w_init[:,np.newaxis],ax=0)
            #print(log_w_prev[:,np.newaxis].shape)
            mini_w=norm_logweights(Log_trans+log_w_init[:,np.newaxis]+log_w_prev[:,np.newaxis],ax=0)
            #print("mini_w is ",mini_w)
            f_LambdaA=np.einsum("ji,jp->ip",mini_w,f_Lambda+term1A)
            f_LambdaB=term1B +term3_Lambd
            f_Lambda= f_LambdaA+f_LambdaB
            
            F_Lambda[i+1]=f_Lambda
            #print("f_Lambda is ",f_Lambda)
            #print("dims are",mini_w.shape,f_b.shape,term2A.shape)
            f_bA=np.einsum("ji,jp->ip",mini_w,f_b+term2A)
            #print("mini_w shape is",mini_w.shape)
            #print("f_b shape is",f_b.shape)
            #print("term2A shape is",term2A.shape)
            #print("term2B shape is",term2B.shape)   
            f_bB=term2B

            f_b=f_bA+f_bB
            #print("f_b shape is ",f_b.shape)
            F_b[i+1]=f_b
            #print("f_b is ",f_b)
            f_g_denA=np.einsum("ji,jpq->ipq",mini_w,f_g_den)
            f_g_denB=term3_g_den
            f_g_den=f_g_denA+f_g_denB
            F_g_den[i+1]=f_g_den
            #print("f_g_den is ",f_g_den)

        if log_w_init.ndim==2:
            #print("log_w_init",log_w_init)
            mini_w=norm_logweights(Log_trans+log_w_init+log_w_prev[:,np.newaxis],ax=0)
            #print("mini_w is ",mini_w)
            f_Lambda=np.einsum("ji,jip->ip",mini_w,\
            (f_Lambda+term1A)[:,np.newaxis]+term1B+term3_Lambd)
            #print( "vec is",(term1A)[:,np.newaxis]+term1B)
            
            F_Lambda[i+1]=f_Lambda
            #print("f_Lambda is ",f_Lambda)
            f_bA=np.einsum("ji,jp->ip",mini_w,f_b+term2A)
            f_bB=term2B
            f_b=f_bA+f_bB
            F_b[i+1]=f_b
            #print("f_b is ",f_b)
            f_g_den=np.einsum("ji,jipq->ipq",mini_w,\
            (f_g_den[samp_par])+term3_g_den)
            #print("term3_g_den is ",term3_g_den)
            #print("f_g_den is ",f_g_den)
            F_g_den[i+1]=f_g_den
        #print("the size of f_b is: ",f_b.shape)
        #print(f_Lambda.shape,norm_logweights(log_w)[:,np.newaxis].shape)
        Grads_Lambda[i+1]=np.sum(norm_logweights(log_w)[:,np.newaxis]*f_Lambda,axis=0)
        Grads_g_den[i+1]=np.sum(norm_logweights(log_w)[:,np.newaxis,np.newaxis]*f_g_den,axis=0)
        Grads_b[i+1]=np.sum(norm_logweights(log_w)[:,np.newaxis]*f_b,axis=0)
        #print("Grads_b is ",Grads_b[i+1])
        step_size=step_0*(i+1)**(-beta)/d
        
        #print("step_size is ",step_size)
        Lambda_pars[i+1]=Lambda_pars[i]+step_size*(Grads_Lambda[i+1]-Grads_Lambda[i])*Lamb_step
        g_pars[i+1]=g_pars[i]+(Grads_g_den[i+1]-Grads_g_den[i])*g_step*step_size
        As[i+1]=As[i]+(Grads_b[i+1]-Grads_b[i])*b_step*step_size
        log_weights[i]=log_weights[i]+log_w
        #print("log_w_init is ",log_w_init)
        weights=norm_logweights(log_weights[i],ax=0)
        x_last=xi[-1]
        x_pf[i]=xi[-1]
        ESS=1/np.sum(weights**2)
        if ESS<resamp_coef*N:
            [samp_par,x_new]=multi_samp(weights,N,x_last,dim)
        else:
            x_new=x_last
            if i< int(T/d)-1:
                log_weights[i+1]=log_weights[i]
            samp_par=np.array(range(N))
        log_w_prev=log_weights[i]
    return [log_weights,x_pf,Lambda_pars,g_pars,As]

#%%


#HERE WE TEST THE SGD FOR NLDT

"""
np.random.seed(0)
T=4000
dim=1
dim_o=dim
xin=np.zeros(dim)+0
l=6
collection_input=[]
I=identity(dim).toarray()
A=np.array([[0]])
fi=1
collection_input=[dim, b_ou,A,Sig_nldt,fi]
cov=I*1e0
g_pars=[dim,cov]
g_par=cov
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=2/9
[obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_pars)
print(obs_time,obs,len(obs_time))
times=2**(-l)*np.array(range(int(T*2**l+1)))

plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
plt.legend()
d=2**(0)
resamp_coef=0.8
"""
#%%
#test nldt

"""
Lamb_par_init=Lamb_par+2
g_par_init=g_par+1
b_numb_par=1
Lamb_par_numb=1
N=10
l=6
d=8
resamp_coef=0.8
beta=3/4.
step_0=.3
b_step=0
g_step=30
np.random.seed(3)
results=Cox_PE(T,xin,b_ou,A,Sig_nldt,fi,Grad_b_gbm,b_numb_par,obs,obs_time,l,d,N,\
dim,resamp_coef,g_den,Grad_Log_g_den,g_par_init,Lambda,Grad_Lambda,Grad_Log_Lambda,\
Lamb_par_init,Lamb_par_numb,step_0,beta)
""" 
#[log_weights,x_pf,Lambda_pars,g_pars,As]
#Lambda_pars=np.zeros((int(T/d)+1,Lamb_par_numb))
#Lambda_pars[0]=Lamb_par
#g_pars=np.zeros((int(T/d)+1,dim,dim))
#g_pars[0]=g_par
#As=np.zeros((int(T/d)+1,b_numb_par))
"""
np.savetxt("Observations&data/SGD_nldt_T1000_dp5_N500_l6_betap75_Lambda_pars.txt",results[2],fmt="%f")
np.savetxt("Observations&data/SGD_nldt_T1000_dp5_N500_l6_betap75_g_pars.txt",results[3].flatten(),fmt="%f")
np.savetxt("Observations&data/SGD_nldt_T1000_dp5_N500_l6_betap75_As.txt",results[4],fmt="%f")
"""
#%%

"""
d=8
T=4000
g_pars=np.reshape(np.loadtxt("Observations&data/SGD_nldt_T1000_dp5_N500_l6_betap75_g_pars.txt"),(int(T/d)+1,dim,dim))
Lambda_pars=np.loadtxt("Observations&data/SGD_nldt_T1000_dp5_N500_l6_betap75_Lambda_pars.txt")
mus=np.loadtxt("Observations&data/SGD_nldt_T1000_dp5_N500_l6_betap75_As.txt")
"""
#%%
#Lambda_pars,g_pars,mus=results[2],results[3],results[4]
#%%
"""
fontsize=14
lw=3
print(Lamb_par,Lamb_par_init)
print("g_par",g_par[0,0])
#print(Lambda_pars)
plt.plot(range(int(T/d)+1),Lamb_par+np.zeros(int(T/d)+1),label=r"$\bar{\theta}_\lambda$",ls="--",c="red",lw=lw)
plt.plot(range(int(T/d)+1),Lambda_pars,label=r"$\theta_\lambda$",c="salmon",lw=lw)
plt.xlabel("Iterations",fontsize=fontsize)
#plt.ylabel(r"$\theta_\lambda$")
print(g_par,g_par_init)
#print(g_pars)
plt.plot(range(int(T/d)+1),g_par[0,0]+np.zeros(int(T/d)+1)\
,label=r"$\bar{\theta}_\Sigma$",ls="dashdot",c="blue",lw=lw)
plt.plot(range(int(T/d)+1),g_pars[:,0,0],label=r"$\theta_\Sigma$",c="dodgerblue",lw=lw)
#print(mu,mu_init)
#print(mu)
#plt.plot(range(int(T/d)+1),mus,label=r"$\theta_b$")
#plt.plot(range(int(T/d)+1),mu[0]+np.zeros(int(T/d)+1),label=r"$\bar{\theta}_b$",ls="--")
plt.legend(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.title("Nonlinear diffusion term",fontsize=fontsize)
#plt.savefig("Images/SGD_nldt_T1000_dp5_N500_l6_betap75_paper_v2.pdf")
"""
#%%
#%%
# NOW WE TEST THE SGD ALGORITHM
"""
np.random.seed(1)
T=4000
dim=1
dim_o=dim
xin=np.zeros(dim)+1
l=8
resamp_coef=0.8
collection_input=[]
I=identity(dim).toarray()
#comp_matrix = ortho_group.rvs(dim)
comp_matrix=np.array([[1]])
inv_mat=la.inv(comp_matrix)
#S=diags(np.random.normal(1,0.1,dim),0).toarray()
S=diags(np.random.normal(0.99,0.1,dim),0).toarray()
#S=np.array([[1.]])
fi=inv_mat@S@comp_matrix
#B=diags(np.random.normal(-1,0.1,dim),0).toarray()
B=diags(np.random.normal(-0.98,0.01,dim),0).toarray()
B=inv_mat@B@comp_matrix
print(B)
#B=np.array([[0.8]])
B_true=B
B_init=B_true+0.5

#B=np.array([[-1.]])
#print(B)
#print(B)
#B=comp_matrix-comp_matrix.T  +B 
collection_input=[dim, b_ou,B,Sig_ou,fi]
cov=I*1e0
g_pars=[dim,cov]
g_par=cov
g_par_init=cov+0.5
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=3.5
Lamb_par_init=Lamb_par-2

[obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_pars)
print(obs_time,obs,len(obs_time))
#plt.plot(times,x_true)
times=2**(-l)*np.array(range(int(T*2**l+1)))
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
#plt.plot(times,kbf,label="KBF")
#plt.plot(times,kf,label="KF")
#plt.plot(times,kf-kbf)
#print(len(obs_time))
plt.legend()
"""
#%%
"""
#PE test
N=2000
l=8
d=2
resamp_coef=0.8
beta=2.5/4.
step_0=1
b_numb_par=1
Lamb_par_numb=1
step_0=2
b_step=1
g_step=0.5

results=Cox_PE(T,xin,b_ou,B_init,Sig_ou,fi,Grad_b_ou,b_numb_par, obs,obs_time,l,d,N,dim,resamp_coef,g_den,Grad_Log_g_den,g_par_init,\
Lambda,Grad_Lambda,Grad_Log_Lambda, Lamb_par_init,Lamb_par_numb,step_0,beta)
"""

#%%
"""
np.savetxt("Observations&data/SGD_ou_T5000_d2_N200_l8_Lambda_pars.txt",results[2],fmt="%f")
np.savetxt("Observations&data/SGD_ou_T5000_d2_N200_l8_g_pars.txt",results[3].flatten(),fmt="%f")
np.savetxt("Observations&data/SGD_ou_T5000_d2_N200_l8_As.txt",results[4],fmt="%f")
"""
#%%

"""
T=4000
d=2
g_pars=np.reshape(np.loadtxt("Observations&data/SGD_ou_T5000_d2_N200_l8_g_pars.txt"),(int(T/d)+1,dim,dim))
Lambda_pars=np.loadtxt("Observations&data/SGD_ou_T5000_d2_N200_l8_Lambda_pars.txt")
As=np.loadtxt("Observations&data/SGD_ou_T5000_d2_N200_l8_As.txt")
"""
#%%
"""
d=2
#Lambda_pars,g_pars,As=results[2],results[3],results[4]
#print(Lamb_par,Lamb_par_init)
#print(Lambda_pars)
plt.plot(range(int(T/d)+1),Lambda_pars,label=r"$\theta_\lambda$",c="coral",lw=3)
plt.plot(range(int(T/d)+1),3.5+np.zeros(int(T/d)+1),label=r"$\bar{\theta}_\lambda$",ls="--",c="red",lw=3)
plt.xlabel("Iterations",fontsize=14)
#plt.ylabel(r"$\theta_\lambda$")
print(g_par,g_par_init)
print(g_pars)
plt.plot(range(int(T/d)+1),g_pars[:,0,0],label=r"$\theta_\Sigma$",c="deepskyblue",lw=3)
plt.plot(range(int(T/d)+1),1+np.zeros(int(T/d)+1),label=r"$\bar{\theta}_\Sigma$",ls="dashdot",c="blue",lw=3)
print(B_true,B_init)
print(As.shape)
plt.plot(range(int(T/d)+1),As[:],label=r"$\theta_b$",c="mediumturquoise",lw=3)
plt.plot(range(int(T/d)+1),-1+np.zeros(int(T/d)+1),label=r"$\bar{\theta}_b$",ls="dotted",c="green",lw=3)
plt.title("Ornstein-Uhlenbeck process",fontsize=14)
plt.legend( ncol=3,loc='upper center',bbox_to_anchor=(0.5, 0.8),fontsize=14)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14 )
plt.locator_params(axis='x', nbins=6)
plt.savefig("images/SGD_ou_T5000_d2_N200_l8_paper_v2.pdf")
"""
#%%

# NOW WE TEST THE SGD ALGORITHM FOR THE LANGEVIN
"""
np.random.seed(1)
T=10000
dim=1
dim_o=dim
xin=np.zeros(dim)+1
l=10
collection_input=[]
I=identity(dim).toarray()
df=10
fi=np.array([[1]])
collection_input=[dim, b_lan,df,Sig_ou,fi]
cov=I*1
g_pars=[dim,cov]
g_par=cov
x_true=gen_gen_data(T,xin,l,collection_input)
Lamb_par=1
[obs_time,obs]=gen_obs(x_true,l,T,Lambda,Lamb_par,g_normal,g_pars)
print(obs_time,obs,len(obs_time))
#plt.plot(times,x_true)
times=2**(-l)*np.array(range(int(T*2**l+1)))
plt.plot(times,x_true,label="True signal")
plt.plot(obs_time,obs,label="Observations")
#plt.plot(times,kbf,label="KBF")
#plt.plot(times,kf,label="KF")
#plt.plot(times,kf-kbf)
#print(len(obs_time))
plt.legend()
"""
#%%
"""
np.savetxt("Observations&data/obs_lan_T10000.txt",obs,fmt="%f")
np.savetxt("Observations&data/obs_time_lan_T10000.txt",obs_time,fmt="%f")
np.savetxt("Observations&data/x_true_time_lan_T10000.txt",x_true,fmt="%f")

"""
#%%
"""
#PE test
N=1000
l=8
d=2
resamp_coef=0.8
beta=3/4.
b_numb_par=1
Lamb_par_numb=1
step_0=2e-1
Lamb_step=1e1 
b_step=0
g_step=1e1
print(df)
df_init=np.array([df])
Lamb_par_init=Lamb_par*2
g_par_init=cov+1.5
results=Cox_PE(T,xin,b_lan,df_init,Sig_ou,fi,Grad_b_lan,b_numb_par, obs,obs_time,l,d,N,dim,resamp_coef,g_den,Grad_Log_g_den,g_par_init,\
Lambda,Grad_Lambda,Grad_Log_Lambda, Lamb_par_init,Lamb_par_numb,step_0,beta)
"""
#%%
"""
Lambda_pars,g_pars,As=results[2],results[3],results[4]
print(As[-1])
#print(Lamb_par,Lamb_par_init)
#print(Lambda_pars)
plt.plot(range(int(T/d)+1),Lambda_pars,label=r"$\theta_\lambda$",c="coral",lw=3)
plt.plot(range(int(T/d)+1),Lamb_par+np.zeros(int(T/d)+1),label=r"$\bar{\theta}_\lambda$",ls="--",c="red",lw=3)
plt.xlabel("Iterations",fontsize=14)
#plt.ylabel(r"$\theta_\lambda$")
print(g_par,g_par_init)
#print(g_pars)
plt.plot(range(int(T/d)+1),g_pars[:,0,0],label=r"$\theta_\Sigma$",c="deepskyblue",lw=3)
plt.plot(range(int(T/d)+1),cov[0,0]+np.zeros(int(T/d)+1),label=r"$\bar{\theta}_\Sigma$",ls="dashdot",c="blue",lw=3)
#print(As)
#plt.plot(range(int(T/d)+1),As[:,0],label=r"$\theta_b$",c="mediumturquoise")
#plt.plot(range(int(T/d)+1),df+np.zeros(int(T/d)+1),label=r"$\bar{\theta}_b$",ls="dotted",c="green")
plt.title("Langevin process",fontsize=14)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.locator_params(axis='x', nbins=6)
plt.legend(fontsize=14)
#plt.legend( ncol=3,loc='upper center',bbox_to_anchor=(0.5, 0.8))
#plt.yscale("log")
plt.savefig("images/SGD_lan_T5000_d2_N1000_l8_paper_v2.pdf")
"""
#%%
"""
np.savetxt("Observations&data/SGD_lan_T1000_d2_N1000_l8_Lambda_pars.txt",results[2],fmt="%f")
np.savetxt("Observations&data/SGD_lan_T1000_d2_N1000_l8_g_pars.txt",results[3].flatten(),fmt="%f")
np.savetxt("Observations&data/SGD_lan_T1000_d2_N1000_l8_As.txt",results[4],fmt="%f")
"""
#%%
"""
d=2
g_pars=np.reshape(np.loadtxt("Observations&data/SGD_lan_T1000_d2_N1000_l8_Lambda_pars.txt"),(int(T/d)+1,dim,dim))
Lamb_par=np.loadtxt("Observations&data/SGD_lan_T1000_d2_N1000_l8_g_pars.txt")
As=np.loadtxt("Observations&data/SGD_lan_T1000_d2_N1000_l8_As.txt")
"""
#%%
# Since it is difficult if not impossible to compare the computation of the gradient a careful check was 
# performed to make sure the gradient is computing the right estimators.


#%%

#%%
#%%
"""
T=10
d=1
obs_time=np.array([0.1,1.,2.,2.01,3.])
c_indices=np.digitize(obs_time,d*np.array(range(int(T/d)+1)),right=True)-1
print(c_indices)
"""
#%%
"""
#This function is not finished, it is supposed to store the original paths 
#and the resampled paths.

def PF(T,z,lmax,x0,b_ou,A,Sig_ou,fi,ht,H,l,d,N,dim,para=True):
    # The particle filter function is inspired in 
    # Bain, A., Crisan, D.: Fundamentals of Stochastic Filtering. Springer,
    # New York (2009).
    # ARGUMENTS: T: final time of the propagation, T>0 and preferably integer
    # z: observation process, its a rank two array with discretized observation
    # at the intervals 2^(-lmax)i, i \in {0,1,...,T2^lmax}. with dimension
    # (T2^{lmax}+1) X dim
    # lmax: level of discretization of the observations
    # x0: initial condition of the particle filter, rank 1 array of dimension
    # dim
    # b_out: function that represents the drift of the process (its specifications
    # is already in the document. A is the arguments taht takes
    # Sig_out: function that represents the diffusion of the process (its specifications
    # is already in the document. Its arguments are included in fi.
    # ht: function in the observation process (its specifications
    # is already in the document). Its arguments are included in H.
    # d: time span in which the resampling is computed. d must be a divisor of T.
    # N: number of particles, N \in naturals greater than 1
    # dim: dimension of the problem
    # para: key to wheter compute the paralelization or not.
    # OUTPUTS: x: is the rank 3 array with the resampled particles at times 
    # 2^{-l}*i, i \in {0,1,..., T*2^l}, its dimentions are (2**l*T+1,N,dim)
    # log_weights: logarithm of the weights at times i*d, for i \in {0,1,...,T/d}.
    # it is a rank 2 array with dimensions (int(T/d),N)
    # suma: its the computation of the particle filter for each dimension of the problem
    # its a rank 2 array with dimensions (int(T/d),dim)
    x=np.zeros((2**l*T+1,N,dim))
    z=cut(T,lmax,l,z)
    log_weights=np.zeros((int(T/d),N))                            
    x_new=x0
    x_resamp=np.zeros((2**l*T,N,dim))
    x[0]=x0
    d_steps=int(d*2**l)
    for i in range(int(T/d)):
        xi=M(x_new,b_ou,A,Sig_ou,fi,l,d,N,dim)
        x[i*d_steps:(i+1)*d_steps]=xi[:-1]
     
        zi=z[i*d_steps:(i+1)*d_steps+1]
        log_weights[i]=G(zi,xi[:-1],ht,H,l,para=True)
        weights=norm_logweights(log_weights[i],ax=0)
        #seed_val=i
        #print(weights.shape)
        x_resamp=sr(weights,N,xi[:-1],dim)[1]
    #Filter
    spots=np.arange(d_steps,2**l*T+1,d_steps,dtype=int)
    x_pf=x[spots]
    weights=norm_logweights(log_weights,ax=1)
    #print(x_pf.shape,weights.shape)
    suma=np.sum(x_pf[:,:,1]*weights,axis=1)
    
    return [x,log_weights,suma]
"""

      
#%%

"""      
l=5
d=1./2**4
N=5
T=10
dim=3
dim_o=dim
#x=M(x0,b_ou,Sig_ou,l,d,N,dim)
I=identity(dim).toarray()
I_o=identity(dim_o).toarray()
#R2=(identity(dim_o).toarray() + np.tri(dim_o,dim_o,1) - np.tri(dim_o,dim_o,-2))/20
R2=I
np.random.seed(0)
H=rand(dim_o,dim,density=0.75).toarray()/1e-2

x0=np.random.normal(1,0,dim)
np.random.seed(3)
comp_matrix = ortho_group.rvs(dim)
inv_mat=la.inv(comp_matrix)
S=diags(np.random.normal(1,0.001,dim),0).toarray()
fi=inv_mat@S@comp_matrix

B=diags(np.random.normal(-.1,0.001,dim),0).toarray()
A=inv_mat@B@comp_matrix

#A=b_ou(I,B).T
R1=Sig_ou(np.zeros(dim),fi)

np.random.seed(3)
C0=I*1e-6
m0=np.random.multivariate_normal(x0,C0)
collection_input=[dim,dim_o,A,R1,R2,H,m0,C0]


[z,x_true]=gen_data(T,l,collection_input)
z=np.reshape(z,z.shape[:2])


[x,log_weights,x_pf]= PF(T,z,l,x0,b_ou,A,Sig_ou,R1,ht,H,l,d,N,dim,para=False)
"""

#%%
"""
lmax=l
a=1
d_steps=int(d*2**l)
spots=np.arange(d_steps,2**l*T+1,d_steps,dtype=int)
z=np.reshape(z,(2**l*T+1,dim,1))
times=np.array(range(int(2**l*T+1)))/2**l



kbf=KBF(T,l,lmax,z,collection_input)
weights=norm_logweights(log_weights,ax=1)
print(x_pf[:,:,a].shape)
xmean=np.sum(weights*x_pf[:,:,a],axis=1)
#plt.plot(times,z[:,a,0])
#plt.plot(times,x_true[:,a,0],label="True signal")
plt.plot(spots/2**l,kbf[0][spots,a,0],label="KBF")
#plt.plot(2**(-l)*np.array(range(T*2**l+1)),kbf[0][:,0,0])
plt.plot(spots/2**l,xmean,label="PF")
#plt.plot(times,x[:,:,a])
#plt.plot(times,xmean,label="mean of the propagation")

plt.legend()
"""
#%%
"""
a=np.array([0,1,2])
print(a.shape)
"""
#%%

"""
print(log_weights)
w=norm_logweights(log_weights[-1],ax=0)
w_total=norm_logweights(log_weights,ax=1)
print(w_total)

print(w)
print(x[-1])
seed_val=0
[part, x_new]=sr(w,N,seed_val,x[-1],dim)
print([part, x_new])

"""


#%%

