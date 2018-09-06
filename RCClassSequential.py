# Created By Parker Kain
# Goal: Learn simple sequential task using Python classes.

#Load libraries
import numpy as np

#----------------------------------------------------------------------------
#Define RC Class (numNeurons not implemented)
class RC:
    def __init__(self, numNeurons, numInputs):
        
        #Set Initial State
        self.x = np.array([[0],[0]])
        self.states = []
        self.numInputs = numInputs
        
        #Set Weights and Biases (random)
        self.W_i2r = np.random.rand(1,2) #Input to Reservoir
        self.W_r2r = np.random.rand(2,2) #Reservoir to Reservoir
        self.W_b2r = np.random.rand(2,1) #Bias to Reservoir
    
    def update(self, u):        
        self.x = np.tanh((self.W_r2r.T @ self.x) + (self.W_i2r.T @ u) + (self.W_b2r))
        self.states.append(self.x)
        return(self.x)
    
    def appendBias(self):
        ones = np.array([1 for i in range(self.numInputs)]).reshape(self.numInputs, 1)
        self.appendedStates = np.append(self.getAllStates(), ones, axis = 1)
        return(self.appendedStates)
        
    def printRC(self):
        print('Current Input to Reservoir Weights ... ')
        print(self.W_i2r, '\n')
        
        print('Current Reservoir to Reservoir Weights ... ')
        print(self.W_r2r, '\n')
        
        print('Current Bias to Reservoir Weights ... ')
        print(self.W_b2r, '\n')
        
    def clearStates(self):
        self.states = []
        
    def getPInv(self):
        self.pInv = np.linalg.pinv(self.appendedStates)
        return(self.pInv.reshape(self.numInputs,3))
        
    def getCurrentState(self):
        return(self.x)
    
    def getAllStates(self):
        return(np.array(self.states).reshape(self.numInputs,2))

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
        
    def getOutput(self, x):
        y = (self.W_r2o.T @ x) + (self.W_b2o)
        return(y)
    
    def updateWeights(self, w):
        self.W_r2o = w
        
    def getR2O(self):
        return(self.W_r2o)
    
    def getB2O(self):
        return(self.W_b2o)
        
    def printReadout(self):
        print('Current Reservoir to Output Weights ...')
        print(self.W_r2o, '\n')
        
        print('Current Bias to Output Weights ...')
        print(self.W_b2o, '\n')
        
#----------------------------------------------------------------------------
class Input:
    def __init__(self, numInputs):
        (self.u, self.y) = createData(numInputs)            
        self.inputCounter = 0
        self.numInputs = numInputs
        
    def getNextU(self):
        nextU = self.u[self.inputCounter]
        self.inputCounter += 1
        
        return(np.array([nextU])) 
        
    def getNext(self):
        nextU = self.u[self.inputCounter]
        nextY = self.y[self.inputCounter]
        self.inputCounter += 1
        
        return(np.array([nextU]), np.array([nextY]))
    
    def getU(self):
        return(self.u)
    
    def getY(self):
        return(self.y)
    
    def getLast5(self):
        print('Last 5 inputs and outputs ...')
    
        for i in range(5):
            print('Input:\n', self.u[self.numInputs-i-10 : self.numInputs-i].reshape(1, 10))
            print('Output:', self.y[self.numInputs-i-1], '\n')
    
#----------------------------------------------------------------------------
def createData(length):
    #Create binary strings, and a count of if there are more 1's in the last 10 timesteps
    u = np.random.randint(2, size=length).reshape(length,1)
    y = []
    
    for i in range(length):
        if i < 10:
            num1 = sum(u[0:i+1])
            num0 = (i+1) - num1
            y.append(int(num1 >= num0))
        else:
            num1 = sum(u[i-9:i+1])
            num0 = 10 - num1
            y.append(int(num1 > num0))
    
    return(u, np.array(y))

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
    my_Data = Input(numInputs)
    my_Data.getLast5()
    
    #Create reservoir
    print('----------------------------------------------------')
    print('Creating Reservoir ... ( Neurons:', numRCNeurons,')\n')
    my_RC = RC(numRCNeurons, numInputs)
    my_RC.printRC()

    #Create Readout
    print('----------------------------------------------------')
    print('Creating Readout ... ( Outputs:',numOutputs,')\n')
    my_Readout = Readout(numOutputs)
    my_Readout.printReadout()
            
    #Loop Passes    
    outputs = []
    my_RC.clearStates()
    for i in range(numInputs):
        nextInput = my_Data.getNextU()
        currentState = my_RC.update(nextInput)
        currentOutput = my_Readout.getOutput(currentState)
        outputs.append(currentOutput[0])

        if verbose:
            print('----------------------------------------------------')
            print('Timestep:', i, '\n')
            print('Current Input:\n', nextInput, '\n')
            print('Current State:\n', currentState,'\n')
            print('Current Output:\n', currentOutput, '\n')
    
    #Adjust Weight Matrix
    print('----------------------------------------------------')  
    my_RC.appendBias()
    mp = my_RC.getPInv()
    outputs = np.array(outputs)

    if verbose:
        print('All States:\n', my_RC.getAllStates(),'\n')
        print('Appended:\n', my_RC.appendBias(), '\n')
        print('Moore Penrose:\n',mp, '\n')
        print('Outputs:\n', outputs, '\n')

    
    print('MP * Outputs')
    mpResults = np.matmul(mp.T, outputs)
    print(mpResults)
    
    my_Readout.updateWeights(mpResults)
    
#----------------------------------------------------------------------------
    
main(verbose = True)