# Written By Parker Kain
# Goal: Use numpy to create a basic reservoir system without learning.

import numpy as np

# ----------------- CREATE MATRICIES ---------------------------

#Create input vector (a single number)
print('Creating input array ...')
x = np.array([1])
print(x, '\n')


#Create reservoir weight matricies
print('Creating reservoir weight matricies ...')
W_i2r = np.random.rand(2,1)
W_r2r = np.random.rand(2,2)
W_b2r = np.random.rand(2,1)

print('Input To Reservoir ... ')
print(W_i2r, '\n')
print('Bias to Reservoir ...')
print(W_b2r, '\n')
print('Reservoir to Reservoir ...')
print(W_r2r, '\n')

np.random.rand