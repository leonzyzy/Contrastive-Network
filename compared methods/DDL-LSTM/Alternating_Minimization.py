# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:25:43 2020

@author: niharika-shimona
"""

import torch
from torch.autograd import Variable


from copy import copy
import numpy as np
import time

#Parallel Processing
from joblib import Parallel, delayed

#my libs
from Optimization_Modules import update_basis, train_c, train_LstmAnn, update_constraints
from Helpers import err_compute, stack_const_mats



def alt_min_main(corr,L,B_init,C_init,model_init,D_init,lamb_init,Y,gamma,lambda_1):
    
    "Main iteration module for alternating minization"
    
    #initialize unknowns
    B_old = B_init #basis
    C_old = C_init #temporal coeffs
    model_old = model_init #lstm-ann
    D_old = D_init #temporal constraints
    lamb_old = lamb_init #lagrangians
    
    #params
    num_iter_max = 50
    thresh = 1e-04 #exit thresh
    
    #pre-allocate
    err_out = np.zeros((num_iter_max,1))
    
    for iter in range(num_iter_max):

        # init err
        err_out[iter] = err_compute(corr,L,B_old,C_old,model_old,D_old,lamb_old,Y,gamma,lambda_1)        
        print(" At iteration: %d || Error: %1.3f  "  %(iter, err_out[iter]) )    
            
        epochs = 51
        lr_nn = 0.0001
         
               
        # run one alternating min step      
        [B,C,model,D,lamb] = alt_min_step(corr,L,B_old,C_old,model_old,D_old,lamb_old,Y,gamma,lambda_1,epochs,lr_nn) 
        
        
        # variable updates
        B_old = B
        C_old = C
        D_old = D
        lamb_old = lamb
        model_old = copy(model)
        
        # check exit conditions
        if((iter>5) and( (abs((err_out[iter]-err_out[iter-1])) < thresh)  or (err_out[iter]-err_out[iter-5]>30))):
           
            if(err_out[iter]>err_out[iter-1]): #fail safe              
                print(' Exiting due to increase in function value, at iter ' ,iter, ' Fix convergence- try adjusting learning rates and schedules')       
                
            break

    return B,C,model,D,lamb,err_out,iter
    
    

def alt_min_step(corr,L,B,C,model,D,lamb,Y,gamma,lambda_1,epochs,lr_nn):
    
   "Given the current values of the iterates, performs a single step of alternating minimization"

   ########
   "Basis update"
   
   print('Optimise B ')   
  
   B_upd = update_basis(B,corr,L,C,D,lamb,lambda_1) 

   print(" At final B iteration || Error: %1.3f " %(err_compute(corr,L,B_upd,C,model,D,lamb,Y,gamma,lambda_1)))  


   ########   
   "Dynamic Coefficients Update"

   print('Optimise C ')
  
   t0 = time.time()
   C_upd =  Parallel(n_jobs=8,backend="threading")(delayed(train_c)(model,C[pat_no],Y[pat_no,:],B_upd,D[pat_no],lamb[pat_no],epochs,gamma,lambda_1) for pat_no in range(Y.shape[0]))
     
   print('{} seconds'.format(time.time() - t0)) 
   print(" Step C || Error: %1.3f"  %(err_compute(corr,L,B_upd,C_upd,model,D,lamb,Y,gamma,lambda_1)))    


   ########
   "LSTM-ANN weight update"
   print('Optimise Theta')
    
   [model_upd,loss] = train_LstmAnn(model,C_upd,Y,epochs,lr_nn,gamma)
   
   print(" Step Theta || Error: %1.3f " %(err_compute(corr,L,B_upd,C_upd,model_upd,D,lamb,Y,gamma,lambda_1)))
  
  
    ########
   "Constraint variable updates "
   t0 = time.time()
   
   num_iter_max = 30
   Const_upd = Parallel(n_jobs=8,backend="threading")(delayed(update_constraints)(B_upd,C_upd[pat_no],corr[pat_no],L[pat_no,:,:],D[pat_no],lamb[pat_no],lambda_1,num_iter_max) for pat_no in range(Y.shape[0]))

   [D_upd,lamb_upd] = stack_const_mats(Const_upd)
   print('{} seconds'.format(time.time() - t0))
   
   print(" Step D,lamb || Error: %1.3f " %(err_compute(corr,L,B_upd,C_upd,model_upd,D_upd,lamb_upd,Y,gamma,lambda_1)))

   return B_upd,C_upd,model_upd,D_upd,lamb_upd