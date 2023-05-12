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
# savez_compressed('AllData.npz',df

