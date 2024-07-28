# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 18:33:42 2020

@author: niharika-shimona
"""
#torch libs
import torch
from torch import optim

import numpy as np

#scipy libs
import scipy
from scipy import optimize

def func(params, *args):
                
    #quadratic objective definition
                
    H_n = args[0].numpy()
    f_n = args[1].numpy()
    
    C_n = params
    C_n_T = np.transpose(C_n)
    f_n_T = np.transpose(f_n)

    error_C_n = 0.5*C_n_T.dot(H_n.dot(C_n)) + f_n_T.dot(C_n)

    return error_C_n
    
def func_der(params, *args):
    
    #quadratic jacobian definition
            
    H_n = args[0].detach().numpy()
    f_n = args[1].detach().numpy()
    
    C_n = params

    grad_C_n = np.matmul(H_n,C_n) + f_n

    return grad_C_n
    

def get_input_optimizer(input_C):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS(input_C,lr=0.9)
    return optimizer
    
    
def Quad_Solve_Coefficients(corr,L,B_gd):

    "Quadratic Solver for coefficients at test time"

    C_upd = []
        
    for pat_no in range(corr.shape[0]): 
        
        for v in range(corr[pat_no].size()[2]):

            Corr_v = corr[pat_no][:,:,v] # dyn correlation matrix		        
            L_v = L[pat_no,:,:] # laplacian
            B_gd_T = B_gd.transpose(0,1) #basis
           
            X =  (B_gd_T.mm(Corr_v.mm(L_v)+L_v.mm(Corr_v))).mm(B_gd)
            
            # quad prog definition
            H_n = 2.0*(B_gd_T.mm(B_gd)).mul((B_gd_T.mm(L_v)).mm(B_gd))           
            f_n = -torch.diag(X)
	
            #initialization
            C_n_init = torch.zeros((B_gd.size()[1],1)).numpy()	
            mybounds = [(1e-8,None)]*np.shape(C_n_init)[0]

            #quadratic solver            
            C_n_upd_test = scipy.optimize.minimize(func,x0=C_n_init,args=(H_n,f_n),method='L-BFGS-B', bounds=mybounds, jac=func_der)		
      
            #create array of temporal coeffs            
            if(v==0):
                 
               m = np.shape(C_n_upd_test.x)[0]
               C_n_upd_test = np.reshape(np.asarray(C_n_upd_test.x),(m,1))
               C_upd_test = torch.from_numpy(C_n_upd_test)
			
            else:
                
               m = np.shape(C_n_upd_test.x)[0]
               C_n_upd_test = np.reshape(np.asarray(C_n_upd_test.x),(m,1))
               C_upd_test = torch.cat((C_upd_test,torch.from_numpy(C_n_upd_test)),1)
         
        #list of patient coeffs                   
        C_upd.append(C_upd_test.float()) 
        
    return C_upd