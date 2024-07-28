# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:03:27 2020

@author: niharika-shimona
"""

#torch libs
import torch
from torch.autograd import Variable


import numpy as np
from copy import copy

from Helpers import loss_regularize


def update_basis(B,corr,L,C,D,lamb,lambda_1):

   "update the basis term B"
   
   M = torch.zeros(B.size()) #initialize procrustest term
   
   for pat_no in range(corr.shape[0]):    
       
       T = corr[pat_no].size()[2] # no of sliding windows per patient
       
       
       for t in range(corr[pat_no].size()[2]):
           
           
           #patient variables           
           Corr_j = corr[pat_no][:,:,t] #dynamic correlations
           L_j = L[pat_no,:,:] #patient laplacian
           D_j  = D[pat_no][t,:,:] #dyanmic constraint term
           lamb_j = lamb[pat_no][t,:,:] #dyanmic lagrangian term
                     
           #update procrustes term          
           M = M + (Corr_j.mm(L_j)+L_j.mm(Corr_j)).mm(D_j)/T + lambda_1*(D_j).mm(torch.diagflat(C[pat_no][:,t]))/T 
           + lambda_1*(lamb_j.mm(torch.diagflat(C[pat_no][:,t])))/T
   
   [U,S,V] = torch.svd(M) #svd
   B_upd = torch.mm(U,torch.transpose(V,0,1)) #closed form update

   return B_upd
   
        
def train_c(lstm_ann,C_data,Y_data,B,D,lamb,num_epochs,gamma,lambda_1):
        
        "Stochastic ADAM update for temporal coefficients per patient"
        
        criterion = torch.nn.MSELoss()    # mean-squared error for regression
        lstm_ann.eval() #set lstm-ann in eval mode to void updating weights

        C_upd = [] #pre-define list
            
        train_X_dat = Variable(torch.transpose(C_data,0,1)).requires_grad_() #training object
        
        #define optimizer and scheduler
        optimizer = torch.optim.Adam([train_X_dat], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.9, last_epoch=-1)
        
        loss_prev = 0 #initialize loss object
         
        for epoch in range(num_epochs):
                        
                inp_relu = train_X_dat.reshape(1,train_X_dat.size()[0],train_X_dat.size()[1]) #LSTM-ANN input
                
                op_relu = lstm_ann.pann_nl(inp_relu)      #relu pre-filtering          
                
                [outputs,attn] = lstm_ann(op_relu)
                
                optimizer.zero_grad() #clear gradients
                
                # obtain the loss function
                train_Y_dat = torch.from_numpy(np.asarray(Y_data,dtype=np.float32)) #targets
                mask = (train_Y_dat>0).type(torch.FloatTensor) #mask for unknown targets
                loss = gamma*criterion(torch.mul(mask,train_Y_dat),torch.mul(outputs,mask)) + loss_regularize(op_relu,B,D,lamb,lambda_1) #total loss
                
                
                loss.backward(retain_graph=True) #backpropagate
                                                                
                # check exit condition                                                                
                if(epoch>10 and 1.2*loss_prev < loss):
                
                    break
            
                loss_prev = loss
                
                #optimization step
                optimizer.step()
                scheduler.step()
                
        del optimizer,scheduler #delete optimization object for new patient
            
        C_upd = torch.transpose(lstm_ann.pann_nl(train_X_dat),0,1).detach() #take relu prefiltered outputs
        
        del lstm_ann,loss
        
        return C_upd
        
   
def train_LstmAnn(lstm_ann,C_data,Y_data,num_epochs,learning_rate,gamma):
        
        "Training the LSTM-ANN network"
        
        criterion = torch.nn.MSELoss()    # mean-squared error for regression
        
        #optimizer and scheduler        
        optimizer = torch.optim.Adam(lstm_ann.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.95, last_epoch=-1)
       
        lstm_ann.train() #train mode

        # Train the model
        
        for epoch in range(num_epochs):
            
            loss_epoch = 0 #initialize epoch loss to 0
            
            for pat_no in range(int(Y_data.shape[0])):    
   
                #forward pass
                train_X_dat = torch.transpose(C_data[pat_no],0,1).float()                
                [outputs,attn] = lstm_ann.forward(train_X_dat.reshape(1,train_X_dat.size()[0],train_X_dat.size()[1]))
                
                optimizer.zero_grad() #clear gradients
                
                # obtain the loss function
                train_Y_dat = torch.from_numpy(np.asarray(Y_data[pat_no,:],dtype=np.float32)) #targets                
                mask = (train_Y_dat>0).type(torch.FloatTensor) # mask out unknowns                              
                loss_batch = criterion(torch.mul(train_Y_dat,mask),torch.mul(outputs,mask))

                loss_batch.backward(retain_graph=True) # backpropagate the loss             
                optimizer.step()                # optimization step
                
                loss_epoch = loss_epoch + loss_batch.detach() #update epoch loss
        
            scheduler.step()

            if(epoch%20 == 0): #print loss every 20 epochs
                
                print("Epoch: %d, loss: %1.3f" % (epoch, loss_epoch)) 
            
       
        #copy model
        lstm_ann_upd = copy(lstm_ann)
        del lstm_ann
        
        return lstm_ann_upd,loss_epoch       
        
        
        
def update_constraints(B_upd,C_upd_n,corr_n,L_n,D_n,lamb_n,lambda_1,num_iter_max):
    
    "Augmented Lagrangian onstraint Updates for a single patient n"

    #fix learning rate
    lr1 = 0.001

    # pre-allocate constraint variables for update
    D_upd = torch.zeros(D_n.size())
    lamb_upd = torch.zeros(lamb_n.size())
    
    for t in range(corr_n.size()[2]):
        
         #patient variables at time point t
        
         corr_n_t = corr_n[:,:,t] #dyanmic patient correlation matrix
         lamb_n_t = lamb_n[t,:,:] #dyanmic lagrangian
         C_n_t = C_upd_n[:,t]  #time varying coefficient
         
         for c in range(num_iter_max):
            
             #primal
             A = (2.0*L_n + lambda_1*torch.eye(L_n.shape[0]))
             B = (L_n.mm(corr_n_t) + corr_n_t.mm(L_n)).mm(B_upd) - lambda_1*lamb_n_t + lambda_1*B_upd.mm(torch.diagflat(C_n_t))           
             
             D_n_t = (torch.pinverse(A)).mm(B) #closed form constraint update
             
             #dual
             lamb_n_t = lamb_n_t + (0.75**(c-1))*lr1*(D_n_t - B_upd.mm(torch.diagflat(C_n_t)))
                   
             #check exit condition         
             if (c ==0):
            
                 grad_norm_init = torch.norm(D_n_t - B_upd.mm(torch.diagflat(C_n_t)))
    
             if (torch.norm(D_n_t - B_upd.mm(torch.diagflat(C_n_t)))/grad_norm_init < 1e-02): #check exit condition -gradient based
     
                break
        
         #assign constraint variables at end of update
         D_upd[t,:,:] = D_n_t 
         lamb_upd[t,:,:] = lamb_n_t

    return [D_upd,lamb_upd]
    
    
    
