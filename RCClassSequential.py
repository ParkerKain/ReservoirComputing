# Created By Parker Kain
# Goal: Learn simple sequential task using Python classes.

#Load libraries
import numpy as np

#----------------------------------------------------------------------------
#Define RC Class (numNeurons not implemented)
class RC:
    def __init__(self, numNeurons):
        
        #Set Initial State
        self.x = np.array([[0],[0]])
        
        #Set Weights and Biases (random)
        self.W_i2r = np.random.rand(2,2) #Input to Reservoir
        self.W_r2r = np.random.rand(2,2) #Reservoir to Reservoir
        self.W_b2r = np.random.rand(2,1) #Bias to Reservoir
    
    def update(self, u):
        self.x = np.tanh((self.W_r2r.T @ self.x) + (self.W_i2r.T @ u) + (self.W_b2r))
        return(self.x)
        
    def getState(self):
        return(self.x)

    def getI2R(self):
        return(self.W_i2r)

    def getR2R(self):
        return(self.W_r2r)
        
    def getB2R(self):
        return(self.W_b2r)
    
#----------------------------------------------------------------------------    
    
class Readout:
    def __init__(self, numNeurons):
        
        #Set Initial Weights
        self.W_r2o = np.random.rand(2,1) #Reservoir to Output
        self.W_b2o = np.random.rand(1,1) #Output to Reservoir
        
    def getR2O(self):
        return(self.W_r2o)
    
    def getB2O(self):
        return(self.W_b2o)
        
#----------------------------------------------------------------------------
class Input:
    def __init__(self, numInputs):
        (self.u, self.y) = createData(numInputs)            
        self.inputCounter = 0
        self.numInputs = numInputs
        
    def getNext(self):
        nextX = self.u[:,self.inputCounter]
        nextY = self.y[:,self.inputCounter]
        self.inputCounter += 1
        
        return(nextX, nextY)
    
    def getU(self):
        return(self.u)
    
    def getY(self):
        return(self.y)
    
    def getLast5(self):
        print('Last 5 inputs and outputs ...')
    
        for i in range(5):
            print('Input:', self.u[self.numInputs-i-10 : self.numInputs-i])
            print('Output:', self.u[self.numInputs-i-1], '\n')
    
#----------------------------------------------------------------------------
def createData(length):
    #Create binary strings, and a count of if there are more 1's in the last 10 timesteps
    x = np.random.randint(2, size=length)
    y = []
    
    for i in range(length):
        if i < 10:
            num1 = sum(x[0:i+1])
            num0 = (i+1) - num1
            y.append(int(num1 >= num0))
        else:
            num1 = sum(x[i-9:i+1])
            num0 = 10 - num1
            y.append(int(num1 > num0))
    
    return(x, np.array(y))

#----------------------------------------------------------------------------
def main(verbose):
    
    #Set seed
    np.random.seed(10)
    
    #Set hyperparameters
    numInputs = 20
    numRCNeurons = 2
    numOutputs = 1
    
    #Create u and y
    print('Creating Data ... ( length =', numInputs, ')\n')
    #(x, y) = createData(20)
    data = Input(numInputs)
    print(data.getLast5())
    
    #Create reservoir
    print('----------------------------------------------------')
    print('Creating Reservoir ... ( Neurons:', numRCNeurons,')\n')
    my_RC = RC(numRCNeurons)
    
    if(verbose):
        print('Initial Input to Reservoir Weights ... ')
        print(my_RC.getI2R(), '\n')
        
        print('Initial Reservoir to Reservoir Weights ... ')
        print(my_RC.getR2R(), '\n')
        
        print('Initial Bias to Reservoir Weights ... ')
        print(my_RC.getB2R(), '\n')
    

    #Create Readout
    print('----------------------------------------------------')
    print('Creating Readout ... ( Outputs:',numOutputs,')\n')
    my_Readout = Readout(numOutputs)
    
    if(verbose):
        print('Initial Reservoir to Output Weights ...')
        print(my_Readout.getR2O(), '\n')
        
        print('Initial Bias to Output Weights ...')
        print(my_Readout.getB2O(), '\n')
        
    #Loop Passes
#----------------------------------------------------------------------------
    
main(verbose = True)