# Created By Parker Kain
# Goal: Learn simple sequential task using Python classes.

#Load libraries
import numpy as np

#----------------------------------------------------------------------------

class Input:
    def __init__(self, numInputs, problem):
        """Initializes input patterns and their correct output, with the inputs generated
        from the createData() function below
        
        numInputs - number of inputs to generate by the createData() function
        """
        
        self.problem = problem
        self.createData(numInputs, problem)            
        self.inputCounter = 0
        self.numInputs = numInputs
        
    def nextU(self):
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
        
    def assessAccuracy(self, predicted):
        """Takes the predicted outputs from the Readout class and compared them to 
        the known outputs
        
        predicted - numpy array of predicted outputs from Readout.getOutput()
        """
    
        reshapedCorrect = self.y.reshape(self.numInputs,1) 
        mse = np.round(((predicted - reshapedCorrect) ** 2).mean(0),4)
        
        boundedPredicted = []
        for i in np.nditer(predicted):
            if i >= 0.5:
                boundedPredicted.append(1)
            else:
                boundedPredicted.append(0)
        boundedPredicted = np.array(boundedPredicted).reshape(self.numInputs, 1)

        (unique, counts) = np.unique(boundedPredicted - reshapedCorrect, return_counts = True) 
        dictCounts = dict(zip(unique,counts))
        
        return(mse, round(dictCounts.get(0) / self.numInputs, 4))
        
    def reset(self):
        """Reset the input counter back to 0, to allow for multiple epochs"""
        self.inputCounter = 0
        
    def createData(self, length, problem):
        """Create binary strings, and a count of if there are more 1's in the last 10 timesteps
        
        length - length of the string to create
        """
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
        
        self.u = u
        self.y = np.array(y)
    
    def getU(self):
        """Returns all input patterns"""
        return(self.u)
    
    def getY(self):
        """Returns all expected outputs"""
        return(self.y)
        
    def getNumInputs(self):
        """Returns the number of timesteps in the training pattern"""
        return(self.numInputs)
    
    def getReshapedY(self):
        """Returns y transposed, as linalg.lstsq() requires it"""
        return(self.y.reshape(self.numInputs,1) )
        
    def getProblem(self):
        """Returns the problem the data was generated from"""
        return(self.problem)
    
    def printLast5(self):
        """Returns the last five inputs and outputs, which is used as a demo to see the data"""
        
        print('Last 5 inputs and outputs ...')
    
        for i in range(5):
            print('Input:\n', self.u[self.numInputs-i-10 : self.numInputs-i].reshape(1, 10))
            print('Output:', self.y[self.numInputs-i-1], '\n')

#----------------------------------------------------------------------------
#Define RC Class (numNeurons not implemented)
class RC:
    def __init__(self, numNeurons, numInputs):
        """Creates the reservoir, initializing the weights to random numbers.
        
        numNeurons -- not implemented
        numInputs -- number of input timesteps (int)
        
        """
        
        #Set Initial State
        self.numNeurons = numNeurons
        self.x = np.zeros((self.numNeurons,1))
        self.states = []
        self.numInputs = numInputs
        
        #Set Weights and Biases (random)
        self.W_i2r = np.random.rand(1,self.numNeurons) * 2 - 1 #Input to Reservoir
        self.W_r2r = np.random.rand(self.numNeurons,self.numNeurons) * 2 - 1 #Reservoir to Reservoir
        self.W_b2r = np.random.rand(self.numNeurons,1) * 2 - 1 #Bias to Reservoir
    
    def update(self, u):      
        """Given a new input, passes said input through the reservoir and returns the new state
        
        u - input pattern from the Input class (returned by nextU() method), numpy array
        """
        
        self.x = np.tanh((self.W_r2r.T @ self.x) + (self.W_i2r.T @ u) + (self.W_b2r))
        self.states.append(self.x)
        return(self.x)
    
    def appendBiasBatch(self):
        """To be used after all timesteps have been passes, appends a column of 1's 
        to represent the bias for the Moore Penrose pseudo inverse"""
        
        ones = np.array([1 for i in range(self.numInputs)]).reshape(self.numInputs, 1)
        self.appendedStates = np.append(self.getAllStates(), ones, axis = 1)
        return(self.appendedStates)
        
    def appendBiasOnline(self):
        """To be used after each timestep has been passed, appends a single 1 to represent
        the bias."""
        
        one = np.array([[1]])
        self.appendedState = np.append(self.getCurrentState(), one, axis = 0)
        return(self.appendedState)
        
    def printRC(self):
        """Prints current weights of the reservoir, which do not change"""
        
        print('Current State ...')
        print(self.x, '\n')
        
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
        self.x = np.zeros((self.numNeurons,1))
        
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
        
        return(np.array(self.states).reshape(self.numInputs, self.numNeurons))

    def getI2R(self):
        """Returns the input to reservoir weight matrix"""
        
        return(self.W_i2r)

    def getR2R(self):
        """Returns the reservoir to reservoir weight matrix"""
        
        return(self.W_r2r)
        
    def getB2R(self):
        """Returns the bias to reservoir weight matrix"""
        
        return(self.W_b2r)
        
    def getNumNeurons(self):
        """Returns the number of neurons in the reservoir """
        return(self.numNeurons)
    
#----------------------------------------------------------------------------    
    
class Readout:
    def __init__(self, numNeurons, numTimesteps, epsilon):
        """Creates the readout layer, initializing the weights and bias to random numbers
        
        numNeurons -- not Implemented
        """
        
        #Set Initial Weights
        self.numNeurons = numNeurons
        self.W_r2o = np.random.rand(numNeurons, 1) * 2 - 1 #Reservoir to Output
        self.W_b2o = np.random.rand(1, 1) * 2 - 1 #Output to Reservoir
        
        #Set some storage for outputs
        self.outputs = []
        self.numTimesteps = numTimesteps
    
        #Set learning rate (for online)
        self.epsilon = epsilon
        
    def getOutput(self, x):
        """Takes a state from the RC class and returns the 
        output after passing it through the readout layer
        
        x - states from reservoir (numpy array)
        """
        
        y = (self.W_r2o.T @ x) + (self.W_b2o)
        self.outputs.append(y)
        return(y)
    
    def updateWeights(self, w, b):
        """Accepts a new weight matrix and bias and updates these values
        
        w - new weight matrix (numpy array)
        b - new bias (numpy array)
        """

        self.W_r2o = w
        self.W_b2o = b        
        
    def gradientUpdate(self, w, b):
        """Updates the Readout layer weights and bias by a factor of the gradient
        
        w - change to weight matrix (numpy array)
        b - change to bias (number)
        """
        
        self.W_r2o -= self.epsilon * w
        self.W_b2o -= self.epsilon * b
        
    
    def getR2O(self):
        """Returns the current reservoir to output weight matrix"""
        
        return(self.W_r2o)
    
    def getB2O(self):
        """Returns the current bias to the output"""
        
        return(self.W_b2o)
    
    def getAllOutputs(self):
        """Returns all appended outputs"""
        
        return(np.array(self.outputs).reshape(self.numTimesteps,1))
    
    def clearReadout(self):
        """Clear out output list"""
        self.outputs = []
        
    def printReadout(self):
        """Prints all weights and biases related to the readout layer"""
        
        print('Current Reservoir to Output Weights ...')
        print(self.W_r2o, '\n')
        
        print('Current Bias to Output Weights ...')
        print(self.W_b2o, '\n')
        
#----------------------------------------------------------------------------
        
def boundOutput(currentOutput):
    """Takes output from the readout layer, and bounds it to either 0 or 1"""
    if currentOutput >= 0.5:
        currentOutput = np.array([[1]])
    else:
        currentOutput = np.array([[0]])
    return(currentOutput)
        
#----------------------------------------------------------------------------
def testLearn(numInputs, problem, my_RC, my_Data, my_Readout):    

    #Test Learning
    my_Test_Data = Input(numInputs, problem)
    
    outputs = []
    my_RC.clearStates()
    my_Data.reset()
    my_Readout.clearReadout()
    
    for i in range(my_Data.getNumInputs()):
        nextInput = my_Test_Data.nextU()
        currentState = my_RC.update(nextInput)
        currentOutput = my_Readout.getOutput(currentState)
        outputs.append(currentOutput[0])
        
    outputs = my_Readout.getAllOutputs()
    (mse, count) = my_Test_Data.assessAccuracy(outputs)
    
    print('Testing MSE:', mse)
    print('Testing Percent Correct:', count)
    
# -------------------------------------------------------------------------
    
def batchLearn(my_Data, my_RC, my_Readout, epochs, verbose):
    """Learns a defined problem using batch learning, rather than online learning
    Will run through the entire training set, and the weights to it"""
    
    #Loop Passes 
    for e in range(2):
        print('----------------------------------------------------')
        print('Beginning Epoch:', e + 1)
        
        #Reset everything
        outputs = []
        my_RC.clearStates()
        my_Data.reset()
        my_Readout.clearReadout()
        
        for i in range(my_Data.getNumInputs()):
            nextInput = my_Data.nextU()
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
        print('----------------------------------------------------')  
        print('Accuracy:\n')

        outputs = my_Readout.getAllOutputs()
        (mse, count) = my_Data.assessAccuracy(outputs)
        print('Training Mean Squared Error:', mse)
        print('Training Percent Correct:', count)


        print('First 20 Expected:\n',my_Data.getY()[0:20])
        print('First 20 Actual:\n',np.round(outputs,0).astype(int)[0:20].reshape(1,20))

        #Adjust Weight Matrix
        print('----------------------------------------------------') 
        
        appended = my_RC.appendBiasBatch()
        new_weights = np.linalg.lstsq(appended, my_Data.getReshapedY(), rcond = None)[0]
    
        if verbose:
            print('All States:\n', my_RC.getAllStates(),'\n')
            print('Appended:\n', my_RC.appendBiasBatch(), '\n')
            print('Outputs:\n', outputs, '\n')
    
        my_Readout.updateWeights(np.array([new_weights[0:my_RC.getNumNeurons()]]).reshape(my_RC.getNumNeurons(), 1), new_weights[my_RC.getNumNeurons()])
    
    return(my_Readout)
#----------------------------------------------------------------------------

def onlineLearn(my_Data, my_RC, my_Readout, epochs, verbose, bound):
    """Learns a defined problem by adjusting the weights of the readout layer
    after each individual timestep"""

    for e in range(epochs):
        print('----------------------------------------------------')
        print('Beginning Epoch:', e + 1)
        
        #Clear all the output streams and reservoir
        outputs = []
        my_RC.clearStates()
        my_Data.reset()
        my_Readout.clearReadout()
        
        for i in range(my_Data.getNumInputs()):
            
            #Grab next timestep
            (nextInput, nextOutput) = my_Data.getNext()
            
            #Pass through network
            currentState = my_RC.update(nextInput)
            currentOutput = my_Readout.getOutput(currentState)
            
            #Bound output between 0 and 1, if wanted
            if bound:
                currentOutput = boundOutput(currentOutput)
            
            outputs.append(currentOutput[0])
            
            #Calculate error and gradient
            error = currentOutput - nextOutput   
            appendedState = my_RC.appendBiasOnline()
            gradient = error * appendedState
            
            #Make updates
            w = gradient[0:my_RC.getNumNeurons(),:]
            b = gradient[my_RC.getNumNeurons(),:]
            my_Readout.gradientUpdate(w, b)
        
        #Assess Accuracy
        outputs = my_Readout.getAllOutputs()
        (mse, count) = my_Data.assessAccuracy(outputs)
        print('Training MSE:', mse)
        print('Training Percent Correct:', count)

        
        #Test accuracy on new data
        testLearn(my_Data.getNumInputs(), my_Data.getProblem(), my_RC, my_Data, my_Readout)


    return(my_Readout)
#----------------------------------------------------------------------------

def main(verbose):
    """Creates and trains a neural net with a reservoir to 
    learn simple binary pattern recognition over time"""
    
    #Set seed
    np.random.seed(10)
    
    #Set hyperparameters
    problem = 'heaviness' #NOT IMPLEMENTED
    numInputs = 100
    numRCNeurons = 5
    numOutputs = 1
    batch = True
    #These only apply to online learning
    learningRate = 0.0001
    epochs = 1000
    bound = False

    
    #Create u and y
    print('Creating Data ... ( length =', numInputs, ')\n')
    my_Data = Input(numInputs, problem)
    my_Data.printLast5()
    
    #Create reservoir
    print('----------------------------------------------------')
    print('Creating Reservoir ... ( Neurons:', numRCNeurons,')\n')
    my_RC = RC(numRCNeurons, numInputs)
    my_RC.printRC()

    #Create Readout
    print('----------------------------------------------------')
    print('Creating Readout ... ( Outputs:',numOutputs,')\n')
    my_Readout = Readout(numRCNeurons, numInputs, learningRate)
    my_Readout.printReadout()
    
    
    if batch:
        #Batch Learn
        my_Readout = batchLearn(my_Data, my_RC, my_Readout, epochs, verbose)
        testLearn(numInputs, problem, my_RC, my_Data, my_Readout)
    else:
        #Online Learn
        my_Readout = onlineLearn(my_Data, my_RC, my_Readout, epochs, verbose, bound)
    
#----------------------------------------------------------------------------
    
main(verbose = False)


#Put in a new pattern ** (Works! Sweet spot for timestep to RC neuron ratio around 40/1)
#Extend the original pattern and see how it learns ** (Concerns about measureing accuracy and overfitting)

# Three tasks:
# "Heaviness task" **
# Label with sequence (robot clamping)
# Parity


#Online learning.
#Training online with gradient descent, see paper. to be used with label to sequence. 