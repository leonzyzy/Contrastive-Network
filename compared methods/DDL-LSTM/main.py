# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:00:34 2020

@author: niharika-shimona
"""

import sys
import numpy as np
import os

# torch
import torch
import pickle

#scipy
import scipy.io as sio


#Matplotlib
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#my libs
from Alternating_Minimization import alt_min_main
from LSTM_ANN import LSTM_ANN
from Helpers import corr_preprocess,init_const
from Quadratic_Solver import Quad_Solve_Coefficients


if __name__ == '__main__':      
    
    #params sr-DDL
    gamma = 3 #regression penalty
    lambda_1 = 20 # constr. tradeoff 
    net = 15 #number of networks 
    
    #params LSTM-ANN
    hidden_size = 40 #hidden layer size
    num_targets = 3 #no of targets
    num_layers = 2 #lstm layers
    input_size = net

    
    path_name = '/home/niharika-shimona/Documents/Projects/Autism_Network/Deep sr-DDL/Data/KKI/'
    output_dirname = path_name + '/Outputs/'

    if not os.path.exists(output_dirname):
            os.makedirs(output_dirname)
    
    #initlialize
    test_pred_Y = []
    test_Y = []
    train_pred_Y = []
    train_Y = []
    
    #print to logfile
    log_filename = output_dirname + 'logfile1.txt'
    log = open(log_filename, 'w')
    sys.stdout = log   
      
    #load data
    data = sio.loadmat(path_name +'/data.mat')

    #train
    corr_train = data['corr_train'][0]
    L_train = torch.from_numpy(data['L_train']).float()
    Y_train = torch.from_numpy(np.asarray(data['Y_train'],dtype=np.float32)).float()
    
    #test
    corr_test = data['corr_test'][0]
    L_test = torch.from_numpy(data['L_test']).float()       
    Y_test = torch.from_numpy(np.asarray(data['Y_test'],dtype=np.float32)).float()
    
    
    ##NOTE: Unknown scores have been set to zero and are not used in estimation of parameters or backpropagation
    
    #e vec sub
    [corr_train,corr_test] = corr_preprocess(corr_train,corr_test)
    
    #initialization
    
    data_init = sio.loadmat(path_name+'/init.mat')
    B_init = torch.from_numpy(data_init['B_gd']).float() #pre-load softstart basis
    C_init = Quad_Solve_Coefficients(corr_train,L_train,B_init) #solve coeffs w/o regression
 
    [D_init,lamb_init] = init_const(B_init,C_init) #const init   
      
    
    model_init = LSTM_ANN(num_targets,input_size,hidden_size,num_layers) #model init

    #run optimization    
    [B_gd,C_gd,model_gd,D_gd,lamb_gd,err_out,iter] = alt_min_main(corr_train,L_train,B_init,C_init,model_init,D_init,lamb_init,Y_train,gamma,lambda_1)
        
    #performance evaluation
   
    #test
    C_test  =  Quad_Solve_Coefficients(corr_test,L_test,B_gd)
    model_gd.eval()
    
    for pat_no in range(corr_test.shape[0]):    
   
        test_X = torch.transpose(C_test[pat_no],0,1).float() #test pat
        outputs = model_gd(test_X.reshape(1,test_X.size()[0],test_X.size()[1]))[0] #forward pass
            
        test_pred_Y.append(outputs.detach().numpy())        
        test_Y.append(Y_test[pat_no,:].detach().numpy())
   
    #train 
   
    for pat_no in range(corr_train.shape[0]):    
   
        train_X = torch.transpose(C_gd[pat_no],0,1).float() #train pat
        outputs = model_gd(train_X.reshape(1,train_X.size()[0],train_X.size()[1]))[0] #forward pass
        
        train_pred_Y.append(outputs.detach().numpy())
        train_Y.append(Y_train[pat_no,:].detach().numpy())
   
    #assign unknown scores to zero
    train_pred_Y[train_Y==0] = 0
    test_pred_Y[test_Y==0] = 0
     
    fig,ax = plt.subplots()
    ax.plot(list(range(iter)),err_out[0:iter],'r')
       
    plt.title('Loss',fontsize=16)
    plt.ylabel('Error' ,fontsize=12)
    plt.xlabel('num of iterations',fontsize=12)
    plt.show()
    figname = output_dirname + 'Loss.png'
    fig.savefig(figname)   # save the figure to fil
    plt.close(fig)
       
    fig1,ax1 = plt.subplots()
    ax1 = plt.imshow(B_gd.detach().numpy(), cmap=plt.cm.jet,aspect='auto')
    plt.title('Recovered Networks')
    plt.show()
    figname1 = output_dirname +'B_gd.png'
    fig1.savefig(figname1,dpi=200)   # save the figure to fil
    plt.close(fig1)
    
    dict_save = {'model': model_gd, 'B_gd': B_gd, 'C_gd': C_gd,'C_test':C_test}
    filename_models =  output_dirname + 'deep_sr-DDL.p'
    pickle.dump(dict_save, open(filename_models, "wb"))
       
    dict_cvf_per = {'Y_test': test_Y, 'Y_pred_train': train_pred_Y, 'Y_train': train_Y, 'Y_pred_test': test_pred_Y}
    sio.savemat(output_dirname+'Performance.mat',dict_cvf_per)
    
    