# -*- coding: utf-8 -*-
import radiomics
import sys, os
import numpy as np
import nibabel as nib
import nrrd
import pandas as pd
from radiomics import featureextractor
from numpy import savez_compressed

# define a function to get each matrix correpsonding to idx
def extractFeature():
    # setup data path
    path = '...'
    os.chdir(path)
    
    # load data
    label = nib.load('...').get_fdata()
    image = nib.load('...').get_fdata() 
    
    # save image into nrrd format
    nrrd.write('image.nrrd', image)
    
    # set up params
    params = '...'
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    
    # define a empty data frame
    features = pd.DataFrame(columns = range(22,122))
    
    # feature extraction 87 ROIs    
    for i in range(1,88):
        print(i)
        mask = (label == i)*1 # get mask
        
        # save mask into nrrd file
        nrrd.write('mask.nrrd', mask)    
        
        # get path
        image_path = path + '/image.nrrd'
        mask_path = path + '/mask.nrrd'

        # get features
        try:
            results = extractor.execute(image_path, mask_path)
            values = list(results.values())[22:122]
              
            features.loc[i] = values
        except ValueError:
            print('Single Voxel, fill 0 instead')
            features.loc[i] = [0]*100
            
     
    # convert data to numpy
    m = features.to_numpy().astype('float64')
  
    return m[np.newaxis,:,:]


extractFeature()
