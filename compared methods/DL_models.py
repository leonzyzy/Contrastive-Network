from keras.layers import Dense, BatchNormalization, Activation, Input, concatenate
#from keras.layers import Concatenate
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D
import os
import random as rn
import tensorflow as tf
rn.seed(12345)
os.environ['PYTHONHASHSEED'] = '0'

            
##################################################################################
          
class DeepMultimodal:
    """
    ======================= Branch =================================
    """
#    @staticmethod
    def build_Connectome_input(inputs, name="Branch_img_input"):  
        branch_input = Flatten()(inputs)       

        branch_input = Dense(64)(branch_input)
        branch_input = BatchNormalization()(branch_input)
        branch_input = Activation('relu')(branch_input) 
        branch_input = Dropout(0.2)(branch_input)
        
        branch_input = Dense(16)(branch_input)
        branch_input = BatchNormalization()(branch_input)
        branch_input = Activation('relu')(branch_input) 
        branch_input = Dropout(0.2)(branch_input)
       
        return branch_input


    def build_Vector_input(inputs, name="Branch_vec_input"):      

        branch_input = Dense(64)(inputs)
        branch_input = BatchNormalization()(branch_input)
        branch_input = Activation('relu')(branch_input) 
        branch_input = Dropout(0.2)(branch_input)
        
        branch_input = Dense(16)(branch_input)
        branch_input = BatchNormalization()(branch_input)
        branch_input = Activation('relu')(branch_input) 
        branch_input = Dropout(0.2)(branch_input)
       
        return branch_input
    """
    =======================multi-input deep CNN model =================================
    """    
#    @staticmethod
    def build(FC_size, chanel_size, SC_size, dwma_len, clin_len):
        
        input1 = Input(shape=(FC_size,FC_size, chanel_size,))   #   
        input2 = Input(shape=(SC_size,SC_size, chanel_size,))   #
        input3 = Input(shape=(dwma_len,))
        input4 = Input(shape=(clin_len,))
        
        ##   Img input feature extraction       
        input1_feat = DeepMultimodal.build_Connectome_input(input1, name="Branch_input1")
        input2_feat = DeepMultimodal.build_Connectome_input(input2, name="Branch_input2")
        input3_feat = DeepMultimodal.build_Vector_input(input3, name="Branch_input3")
        input4_feat = DeepMultimodal.build_Vector_input(input4, name="Branch_input4")
        
        #    merge img data
        hl_feat = concatenate([input1_feat, input2_feat, input3_feat,
                               input4_feat], axis=1)
               
        hl_feat = Dense(8)(hl_feat)
        hl_feat = BatchNormalization()(hl_feat)
        hl_feat = Activation('relu')(hl_feat) 
        
        ##   output
        hl_feat = Dense(2)(hl_feat)
        hl_feat = BatchNormalization()(hl_feat)
        Output = Activation('softmax')(hl_feat)       
       
        ##   combine
        model = Model(inputs=[input1, input2, input3, input4], 
                      outputs = Output,
                      name = "Multimodal")       
        return model
