# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:14:23 2019

@author: kainp1
"""

import glob
import pandas as pd
import numpy as np
 # get data file names
 
def gatherData():
        
    print('Gathering ...')
    
    path =r'C:\Users\kainp1\Documents\GitHub\ReservoirComputing\Parkinsons\hw_dataset\control'
    c_filenames = glob.glob(path + "/*.txt")
    
    path =r'C:\Users\kainp1\Documents\GitHub\ReservoirComputing\Parkinsons\hw_dataset\parkinson'
    p_filenames = glob.glob(path + "/*.txt")
    
    path = r'C:\Users\kainp1\Documents\GitHub\ReservoirComputing\Parkinsons\new_dataset\parkinson'
    h_filenames = glob.glob(path + '/*.txt')
    
    filenames = c_filenames + p_filenames + h_filenames
    
    #Load in each text file
    col_Names=["X", "Y", "Z", "Pressure","GripAngle","Timestamp","ID"]
    dfs = []
    
    for filename in filenames:
        if 'P_' in filename or 'H_' in filename:
            parkinsonsFlag = 1
        else:
            parkinsonsFlag = 0
            
        table = pd.read_table(filename, sep = ';', header = None, names = col_Names)
        table = table[table.ID == 0]
        table = table.assign(Parkinsons=pd.Series([parkinsonsFlag] * len(table)))
        dfs.append(table)
    
    
    #Get longest length so I can zero pad the rest
    lengths = [len(entry) for entry in dfs]
    print(lengths)
    maxLength = max(lengths)
    print(maxLength)
    
    #Perform zero padding
    zeroPadded = []
    for df in dfs:
        parkinsonsFlag = max(df['Parkinsons'])
        for i in range(maxLength):
            if i >= len(df):
                df = df.append({'X' : 0 , 'Y' : 0, 'Z' : 0, 'Pressure' : 0, 
                                'GripAngle' : 0, 'Timestamp' : 0, 'ID' : 0,
                                'Parkinsons' : parkinsonsFlag}, ignore_index = True)
        zeroPadded.append(df)
                
        
    #Convert to numpy array
    numpyToSave = np.array([zeroPadded[i].values for i in range(len(dfs))])
    print(numpyToSave.shape)
    print(type(numpyToSave))
    
    #Split into u_train, y_train, and y_lag_train
    
    numpyToSave = np.transpose(numpyToSave, (1,2,0))
    np.save('parkinsons.npy', numpyToSave)
    
#-------------------------------------------------------------
    
def transformData():
    import mpu.ml

    print('Transforming ...')
    
    parkinsons = np.load('parkinsons.npy')
    
    #Create u_train
    u_train = parkinsons[:,[0,1,3,4],:]    
    
    #Create y_train
    y = parkinsons[:,7,:]
    
    y_list = []
    for pattern in np.transpose(y, (1,0)):
        one_hot = np.array(mpu.ml.indices2one_hot(pattern, nb_classes=2))
        y_list.append(one_hot)

    y_train = np.array(y_list)
    y_train = np.transpose(y_train, (1,2,0))    
    
    #Create y_lag_train
    
    y_lag_train = np.append(np.zeros((1,2,76)), y_train, axis = 0)[:-1]

    print(u_train.shape)
    print(y_train.shape)
    print(y_lag_train.shape)

    print(u_train[:,:,0])
    print(y_train[:,:,0])

    import random
    test_nums = random.sample(range(76), 38)
    print(test_nums)
    
    u_test = u_train[:,:,test_nums]
    y_test = y_train[:,:,test_nums]
    y_lag_test = y_lag_train[:,:,test_nums]
    
    u_train = np.delete(u_train, test_nums, 2)
    y_train = np.delete(y_train, test_nums, 2)
    y_lag_train = np.delete(y_lag_train, test_nums, 2)

    print(u_train.shape)
    
    

    np.save('u_train.npy', u_train)
    np.save('y_train.npy', y_train)
    np.save('y_lag_train.npy', y_lag_train)
    np.save('u_test.npy', u_test)
    np.save('y_test.npy', y_test)
    np.save('y_lag_test.npy', y_lag_test)


#gatherData()
transformData()

#Note, can throw away Z and ID columns