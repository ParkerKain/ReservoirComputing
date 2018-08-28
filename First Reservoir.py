# Written By Parker Kain.
# Goal: Use numpy to create a basic reservoir system without learning.

import numpy as np
np.random.seed(10)

# ----------------- CREATE MATRICIES ---------------------------

#Create input vector (a single number)
print('Creating input array ...')
u = np.array([1])
x = np.array([[0],[0]])
y = 0

print('Input ...')
print(u, '\n')
print('Starting Reservoir State ...')
print(x, '\n')
print('Starting Output ...')
print(y, '\n')


#Create reservoir weight matricies
print('Creating reservoir weight matricies ...')

W_i2r = np.random.rand(2,1) #Input to Reservoir
W_r2r = np.random.rand(2,2) #Reservoir to Reservoir
W_b2r = np.random.rand(2,1) #Bias to Reservoir

print('Input To Reservoir ... ')
print(W_i2r, '\n')
print('Bias to Reservoir ...')
print(W_b2r, '\n')
print('Reservoir to Reservoir ...')
print(W_r2r, '\n')

#Create Output Layer weight matricies
print('Creating output layer weight matricies ...')

W_r2o = np.random.rand(1,1) #Reservoir to Output
W_b2o = np.random.rand(1,1) #Output to Reservoir

print('Reservoir to Output ...')
print(W_r2o, '\n')
print('Bias to Output ...')
print(W_b2o, '\n')


# ---------------------------- RESERVOIR PASSTHROUGH ------------------------
print('Beginning Reservoir Passthrough ... \n')

#Pass through the reservoir

x = (W_i2r * u) + (W_b2r)
print('Current Output ...')
print(x)