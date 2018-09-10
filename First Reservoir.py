# Written By Parker Kain.
# Goal: Use numpy to create a basic reservoir system without learning.

#Load packages
import numpy as np

#----------------------------------------------------------------------------
def main(verbose = True):
    
    #Set seed
    np.random.seed(10)
    
    #Set hyperparameters
    epochs = 3
    
    #Initialize Weight matricies, establish inputs
    (u_tot, x, W_i2r, W_b2r, W_r2r, W_r2o, W_b2o) = init(verbose)
    num_inputs = u_tot.shape[1]
    
    #Begin Epochs
    for e in range(epochs):
        print('--------------------------------------------------------')
        print('                EPOCH NUMBER', e+1)
        print('--------------------------------------------------------')

        if e < num_inputs:
            u = u_tot[:,[e]]
        else:
            u = np.array([[0],[0]])
    
        #Passthrough Reservoir
        x = reservoirPass(u, x, W_i2r, W_r2r, W_b2r, verbose)
        
        #Passthrough Readout
        y = readoutPass(x, W_r2o, W_b2o, verbose)

#---------------------------------------------------------------- ------------    
    
def init(verbose):

    #Create input vector (a single number)
    print('Creating input array ...')
    u_tot = np.array([[1, 2],[1, 2]])
    x = np.array([[0],[0]])
    
    if(verbose):
        print('Input (u_tot)...')
        print(u_tot, '\n')
        
        print('Starting Reservoir State (x)...')
        print(x, '\n')
        
    
    #Create reservoir weight matricies
    print('Creating reservoir weight matricies ...')
    
    W_i2r = np.random.rand(2,2) #Input to Reservoir
    W_r2r = np.random.rand(2,2) #Reservoir to Reservoir
    W_b2r = np.random.rand(2,1) #Bias to Reservoir
    
    if(verbose):
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
    
    if(verbose):
        print('Reservoir to Output (W_r2o)...')
        print(W_r2o, '\n')
        
        print('Bias to Output (W_b2o)...')
        print(W_b2o, '\n')
        print('--------------------------------------------------------')

    return(u_tot, x, W_i2r, W_b2r, W_r2r, W_r2o, W_b2o)
    
#----------------------------------------------------------------------------    
    
def reservoirPass(u, x, W_i2r, W_r2r, W_b2r, verbose):
    
    print('Beginning Reservoir Passthrough ... ')
    #Pass through the reservoir
    oldX = x #Just storing to print if verbose
    x = np.tanh((W_r2r.T @ x) + (W_i2r.T @ u) + (W_b2r))
    
    if(verbose):
        print('Input to Reservoir Part ...')
        print(W_i2r.T @ u, '\n')
        
        print('Res to Res Part ...')
        print(W_r2r.T @ oldX, '\n')
        
        print('Bias to Res Part ...')
        print(W_b2r, '\n')
    
        print('Reservoir Output (after firing)...')
        print(x)
        print('--------------------------------------------------------')

    return(x)
    
#----------------------------------------------------------------------------
    
def readoutPass(x, W_r2o, W_b2o, verbose):
    
    print('Beginning Readout Layer ... ')
    
    y = (W_r2o.T @ x) + (W_b2o)
    
    if(verbose):
        print('Reservoir to Output Part ... ')
        print(W_r2o.T @ x, '\n')
        
        print('Bias to Output Part ...')
        print(W_b2o, '\n')
        
    print('Final Output ...')
    print(y)    

    return(y)    
    
#----------------------
main(verbose = True)
#----------------------
 