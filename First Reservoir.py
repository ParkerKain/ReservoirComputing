# Written By Parker Kain.
# Goal: Use numpy to create a basic reservoir system without learning.

import numpy as np
np.random.seed(10)

# ----------------- CREATE MATRICIES ---------------------------

#Create input vector (a single number)
print('Creating input array ...')
u = np.array([[1],[1]])
x = np.array([[0],[0]])
y = 0

print('Input (u)...')
print(u, '\n')

print('Starting Reservoir State (x)...')
print(x, '\n')

print('Starting Output (y)...')
print(y, '\n')


#Create reservoir weight matricies
print('Creating reservoir weight matricies ...')

W_i2r = np.random.rand(2,2) #Input to Reservoir
W_r2r = np.random.rand(2,2) #Reservoir to Reservoir
W_b2r = np.random.rand(2,1) #Bias to Reservoir

print('Input To Reservoir (W_i2r)... ')
print(W_i2r, '\n')

print('Bias to Reservoir (W_b2r)...')
print(W_b2r, '\n')

print('Reservoir to Reservoir (W_r2r)...')
print(W_r2r, '\n')

#Create Output Layer weight matricies
print('Creating output layer weight matricies ...')

W_r2o = np.random.rand(2,1) #Reservoir to Output
W_b2o = np.random.rand(1,1) #Output to Reservoir

print('Reservoir to Output (W_r2o)...')
print(W_r2o, '\n')

print('Bias to Output (W_b2o)...')
print(W_b2o, '\n')

# ---------------------------- RESERVOIR PASSTHROUGH ------------------------
print('--------------------------------------------------------')
print('Beginning Reservoir Passthrough ... \n')

#Pass through the reservoir

print('Input to Reservoir Part ...')
print(W_i2r.T @ u, '\n')

print('Res to Res Part ...')
print(W_r2r.T @ x, '\n')

print('Bias to Res Part ...')
print(W_b2r, '\n')

x = (W_r2r.T @ x) + (W_i2r.T @ u) + (W_b2r)
print('Reservoir Output ...')
print(x)

# ------------------------ READOUT LAYER PASSTHROUGH -----------------------
print('--------------------------------------------------------')
print('Beginning Readout Layer ... \n')

print('Reservoir to Output Part ... ')
print(W_r2o.T @ x, '\n')

print('Bias to Output Part ...')
print(W_b2o, '\n')

y = (W_r2o.T @ x) + (W_b2o)
print('Final Output ...')
print(y)