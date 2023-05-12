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
def extractFeature(idx):
    # setup data path
    path = 'your path'
    os.chdir(path)
    
    # input your data
    label = nib.load('....').get_fdata()
    image = nib.load('....').get_fdata() 
    
    # save image into nrrd format
    nrrd.write('image.nrrd', image)
    
    # set up params, use your .yaml style
    params = '....'
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

# define a function to genereate data 
def extractAllData():
    # set path
    path = 'F:/Zhiyuan/T2 dhcp/derivatives_corrected'
    os.chdir(path)
    
    # get each subject 
    with open('subject.txt') as f:
        sub_names = [line.strip() for line in f]
    
    # define a data object
    data = np.empty((0,87,100)).astype('float64')
    
    # append samples into data
    for idx in sub_names:
        try:
            m = extractFeature(idx)
            data = np.append(data, m, axis=0)
        except ValueError:
            print('Cannot extract features for a single voxel ({})'.format(idx))
            pass
        print("Adding data (shape): {}".format(data.shape))

    return data

allMeasures = extractAllData()

# save data
nrrd.write('allMeasures.nrrd', allMeasures)

# test for subject 330
m = extractFeature('sub-330')
m = np.nan_to_num(m)
m[:,26,:]

# insert into data
df, head = nrrd.read('allMeasures.nrrd')
df = np.insert(df, 150, m, 0)
# nrrd.write('AllMeasuresData.nrrd', df)

# save as npz file 
# savez_compressed('AllData.npz',df)
