# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:14:23 2019

@author: kainp1
"""

import glob
import pandas as pd
import numpy as np
 # get data file names
path =r'C:\Users\kainp1\Documents\GitHub\ReservoirComputing\Parkinsons\hw_dataset\control'
filenames = glob.glob(path + "/*.txt")

#Load in each text file
col_Names=["X", "Y", "Z", "Pressure","GripAngle","Timestamp","ID"]
dfs = []
for filename in filenames:
    table = pd.read_table(filename, sep = ';', header = None, names = col_Names)
    table = table[table.ID == 0]
    dfs.append(table)


#Get longest length so I can zero pad the rest
lengths = [len(entry) for entry in dfs]
print(lengths)
maxLength = max(lengths)
print(maxLength)

#Perform zero padding
zeroPadded = []
for df in dfs:
    for i in range(maxLength):
        if i >= len(df):
            df = df.append({'X' : 0 , 'Y' : 0, 'Z' : 0, 'Pressure' : 0, 
                            'GripAngle' : 0, 'Timestamp' : 0, 'ID' : 0}, ignore_index = True)
    zeroPadded.append(df)
            
for df in zeroPadded:
    print(df.shape)
    
#Convert to numpy array
numpyToSave = np.array([zeroPadded[i].values for i in range(len(dfs))])
print(numpyToSave.shape)
print(type(numpyToSave))

numpyToSave = np.transpose(numpyToSave, (2,1,0))

np.save('parkinsons.npy', numpyToSave)

#Nore, can throw away Z and ID columns