# Created By Parker Kain
# Goal: Learn simple sequential task using Python classes.

#Load libraries
import numpy as np

#----------------------------------------------------------------------------
#Define RC Class (numNeurons not implemented)
class RC:
    def __init__(self, numNeurons, numInputs):
        """Creates the reservoir, initializing the weights to random numbers."""
        
        #Set Initial State
        self.x = np.array([[0],[0]])
        self.states = []
        self.numInputs = numInputs
        
        #Set Weights and Biases (random)
        self.W_i2r = np.random.rand(1,2) #Input to Reservoir
        self.W_r2r = np.random.rand(2,2) #Reservoir to Reservoir
        self.W_b2r = np.random.rand(2,1) #Bias to Reservoir
    
    def update(self, u):      
        """Given a new input, passes said input through the reservoir and returns the new state"""
        
        self.x = np.tanh((self.W_r2r.T @ self.x) + (self.W_i2r.T @ u) + (self.W_b2r))
        self.states.append(self.x)
        return(self.x)
    
    def appendBias(self):
        """To be used after all timesteps have been passes, appends a column of 1's 
        to represent the bias for the Moore Penrose pseudo inverse"""
        
        ones = np.array([1 for i in range(self.numInputs)]).reshape(self.numInputs, 1)
        self.appendedStates = np.append(self.getAllStates(), ones, axis = 1)
        return(self.appendedStates)
        
    def printRC(self):
        """Prints current weights of the reservoir, which do not change"""
        
        print('Current Input to Reservoir Weights ... ')
        print(self.W_i2r, '\n')
        
        print('Current Reservoir to Reservoir Weights ... ')
        print(self.W_r2r, '\n')
        
        print('Current Bias to Reservoir Weights ... ')
        print(self.W_b2r, '\n')
        
    def clearStates(self):
        """To be used between epochs, clears the reservoir states array as 
            well as the reservoir to reservoir weights"""
            
        self.states = []
        self.x = np.array([[0],[0]])
        
        
    def getPInv(self):
        """Calculates the Moore Penrose Pseudo Inverse for the concatenated and appended 
        reservoir states. Returns the inverse to be used for learning"""
        
        self.pInv = np.linalg.pinv(self.appendedStates)
        return(self.pInv.reshape(self.numInputs,3))
        
    def getCurrentState(self):
        """Returns the current reservoir state."""
        
        return(self.x)
    
    def getAllStates(self):
        """Returns all states so far for the current epoch"""
        
        return(np.array(self.states).reshape(self.numInputs,2))

    def getI2R(self):
        """Returns the input to reservoir weight matrix"""
        
        return(self.W_i2r)

    def getR2R(self):
        """Returns the reservoir to reservoir weight matrix"""
        
        return(self.W_r2r)
        
    def getB2R(self):
        """Returns the bias to reservoir weight matrix"""
        
        return(self.W_b2r)
    
#----------------------------------------------------------------------------    
    
class Readout:
    def __init__(self, numNeurons):
        """Creates the readout layer, initializing the weights and bias to random numbers"""
        
        #Set Initial Weights
        self.W_r2o = np.random.rand(2,1) #Reservoir to Output
        self.W_b2o = np.random.rand(1,1) #Output to Reservoir
        
    def getOutput(self, x):
        """Takes a state from the RC class and returns the 
        output after passing it through the readout layer"""
        
        y = (self.W_r2o.T @ x) + (self.W_b2o)
        return(y)
    
    def updateWeights(self, w, b):
        """Accepts a new weight matrix and bias and updates these values"""
        
        self.W_r2o = w
        self.W_b2o = b
        
    def getR2O(self):
        """Returns the current reservoir to output weight matrix"""
        
        return(self.W_r2o)
    
    def getB2O(self):
        """Returns the current bias to the output"""
        
        return(self.W_b2o)
        
    def printReadout(self):
        """Prints all weights and biases related to the readout layer"""
        
        print('Current Reservoir to Output Weights ...')
        print(self.W_r2o, '\n')
        
        print('Current Bias to Output Weights ...')
        print(self.W_b2o, '\n')
        
#----------------------------------------------------------------------------
class Input:
    def __init__(self, numInputs):
        """Initializes input patterns and their correct output, with the inputs generated
        from the createData() function below"""
        
        (self.u, self.y) = createData(numInputs)            
        self.inputCounter = 0
        self.numInputs = numInputs
        
    def getNextU(self):
        """Passes the next input pattern, which will be used to pass one pattern at a time
        to the reservoir"""
        
        nextU = self.u[self.inputCounter]
        self.inputCounter += 1
        
        return(np.array([nextU])) 
        
    def getNext(self):
        """Passes the next input pattern as well as its corresponding expected output
        which can be used to assess accuracy"""
        nextU = self.u[self.inputCounter]
        nextY = self.y[self.inputCounter]
        self.inputCounter += 1
        
        return(np.array([nextU]), np.array([nextY]))
    
    def getU(self):
        """Returns all input patterns"""
        return(self.u)
    
    def getY(self):
        """Returns all expected outputs"""
        return(self.y)
    
    def getLast5(self):
        """Returns the last five inputs and outputs, which is used as a demo to see the data"""
        
        print('Last 5 inputs and outputs ...')
    
        for i in range(5):
            print('Input:\n', self.u[self.numInputs-i-10 : self.numInputs-i].reshape(1, 10))
            print('Output:', self.y[self.numInputs-i-1], '\n')
    
#----------------------------------------------------------------------------
def createData(length):
    """Create binary strings, and a count of if there are more 1's in the last 10 timesteps"""
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
    """Creates and trains a neural net with a reservoir to learn simple binary pattern 
    recognition over time"""
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
            
    #Assess Accuracy
    
    
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

    my_Readout.updateWeights(mpResults[0:2,:], mpResults[2,:])
    
#----------------------------------------------------------------------------
    
main(verbose = True)