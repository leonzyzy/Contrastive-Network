# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:07:46 2020

@author: niharika-shimona
"""



import torch

import numpy as np


############################################################################### 
"Helper functions for main optimization modules"   

def corr_preprocess(corr_train_np,corr_test_np):

    "convert dynamic correlation matrices to torch objects after eigenvector subtraction"

    #sizes of train and test sets
    N = np.shape(corr_train_np)[0]
    N_test = np.shape(corr_test_np)[0]
    
    #pre-allocate list objects
    corr_train  = corr_train_np
    corr_test = corr_test_np
    
    #train set
    for pat_no_train in range(N):

        corr_mat = corr_train_np[pat_no_train] #extract patient matrices
        corr_dash = torch.from_numpy(corr_mat).float()   #convert to torch tensors 
        corr_train[pat_no_train] = mean_sub_eig(corr_dash) #perform eigen subtraction

    #test set
    for pat_no in range(N_test):

        corr_mat = corr_test_np[pat_no] #extract patient matrices
        corr_dash_test = torch.from_numpy(corr_mat).float() #convert to torch tensors        
        corr_test[pat_no] = mean_sub_eig(corr_dash_test) #perform eigen subtraction

    return corr_train,corr_test

def mean_sub_eig(corr):
    
    " Perform eigenvector subtraction on dynamic correlation matrices"
    
    #pre-allocate
    corr_eig_sub = torch.zeros(corr.size())
    
    for v in range(corr.size()[2]):
        
        corr_mat = corr[:,:,v].reshape(corr.size()[0],corr.size()[0]) #cast as sq-matrix
        
        [D,V] = torch.eig(corr_mat,True) #eig-decomposition
        
        max_value = max(D[:,0]) #max eval

        max_index = list(D[:,0]).index(max_value) #index max evec        
        V_vec = V[:,max_index].reshape(V.size()[0],1) #max evec
        V_T = torch.transpose(V_vec,0,1)
          
        sub_val = max_value*torch.mm(V_vec,V_T) #contribution    
        corr_eig_sub[:,:,v] = corr_mat - sub_val #remove contribution
  
    return corr_eig_sub

def init_const(B_init,C_init):
    
    "Inititalize constraint variables at start"

    #pre-allocate list variables
    D_init = []
    lamb_init = []
    
    for pat_no in range(len(C_init)):
        
        #pre-allocate
        D_init_mat = torch.zeros(C_init[pat_no].size()[1],B_init.size()[0],C_init[pat_no].size()[0])
        lamb_init_mat = torch.zeros(C_init[pat_no].size()[1],B_init.size()[0],C_init[pat_no].size()[0])
        
        for t in range(C_init[pat_no].size()[1]):
        
            D_init_mat[t,:,:] = B_init.mm(torch.diagflat(C_init[pat_no][:,t])) # constraint variable definition for feasible initial solution
        
        D_init.append(D_init_mat)
        lamb_init.append(lamb_init_mat)
        
    return D_init,lamb_init
    
    
def err_compute(corr,L,B,C,model,D,lamb,Y,gamma,lambda_1):
    
    "Computes the error at the current main interation"

    #initialize
    fit_err = 0 # correlation fit
    const_err = 0 # constraint
    aug_lag_err = 0 # aug-lag regularizer
    
    est_Y = torch.zeros(Y.size()) #estimate of scores
    
    for pat_no in range(Y.size()[0]):
         
        #patient variables
        C_pat = torch.transpose(C[pat_no],0,1) #temporal coeffs.
        L_n = L[pat_no,:,:] #laplacian
        T = corr[pat_no].size()[2] # no of sliding windows
        
        for t in range(corr[pat_no].size()[2]):
            
            #patient variables
            Corr_n = corr[pat_no][:,:,t] #dynamic corr. matrix         
            D_n_t = D[pat_no][t,:,:] #dynamic const. 
            lamb_n_t = lamb[pat_no][t,:,:] #dynamic aug. lag.
            
            #correlation fit update
            X = Corr_n - torch.mm(D_n_t,B.transpose(0,1)) 
            fit_err = fit_err + torch.trace(((X.transpose(0,1)).mm(L_n)).mm(X))/T 
            
            #constraint fit update
            lamb_n_t_T   = lamb_n_t.transpose(0,1)         
            const_err = const_err + (1.0/T)*torch.trace(lamb_n_t_T.mm(D_n_t - B.mm(torch.diagflat(C[pat_no][:,t]))))
            
            #aug.lag  update
            aug_lag_err = aug_lag_err + 0.5*(1.0/T)*torch.norm(D_n_t - B.mm(torch.diagflat(C[pat_no][:,t])))**2
        
        #lstm-ann forward pass
        est_Y[pat_no,:] = model.forward(C_pat.reshape(1,C_pat.size()[0],C_pat.size()[1]))[0]           

    #mask unknown targets
    mask = (Y>0).type(torch.FloatTensor)
     
    #total error at main iteration
    err = fit_err + (gamma*torch.norm(torch.mul(est_Y-Y,mask))**2) + lambda_1*(const_err+aug_lag_err) 
    
    return err.detach().numpy()

def loss_regularize(C_relu,B_upd,D_pat,lamb,lambda_1):
    
    "Regularizing the stochastic update for coefficients via joint objective"
    
    reg_err = 0 #initialize loss
    T = D_pat.size()[0] #no of sliding windows per patient
  
    for t in range(D_pat.size()[0]): 
         
         #Patient variables 
         lamb_t = lamb[t,:,:] # dynamic lagrangian term
         lamb_t_T = torch.transpose(lamb_t,0,1)
         D_t = D_pat[t,:,:] #dynamic constraint term
         D_n = B_upd.mm(torch.diagflat(C_relu[:,t])) # constraint argument
           
         #update loss
         reg_err = reg_err + lambda_1*torch.trace(lamb_t_T.mm(D_t - D_n))/T + 0.5*lambda_1*(torch.norm(D_t - D_n)**2)/T
 
    return reg_err

def stack_const_mats(Const_upd):

    "Recast patient dyanmic constraint matrices as lists"
    
    #pre-allocate
    D_upd  = []   
    lamb_upd = []

    for j in range(len(Const_upd)):
        
        D_upd.append(Const_upd[j][0].detach()) #list update- constraint
        lamb_upd.append(Const_upd[j][1].detach()) #list update- lagrangian
       
    return D_upd,lamb_upd