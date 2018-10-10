import numpy as np
from pyESN import ESN

#Set Seed
rng = np.random.RandomState(42)

def sigmoid(a):
    """Sigmoid (logistic) firing function."""
    return 1/(1 + np.exp(-a)) 
    
def dsigmoid(b):
    """Derivative of sigmoid firing function b=fire(a), as function of b."""
    # Used in the backpropagation algorithm to compute error signals at hidden layers.
    return b*(1-b)

def invsigmoid(a):
    return np.log(a/(1-a))

def generateParity(length, parity):
    u = np.random.randint(2, size=length).reshape(length,1)
    y = []
    currentParity = 0 
    
    for i in range(length):
        
        if parity == 2:
            parityState = np.array([0])
        else:
            parityState = np.zeros(parity-1)   
        
        currentU = u[i]
        currentParity = (currentParity + currentU) % parity
        
        parityState = np.zeros((1, parity-1)) -1 
        if currentParity != 0:
            parityState[:,currentParity-1] = 1
        
        parityState = parityState * 0.99
        y.append(parityState)
        
    return(u, np.array(y).reshape(length, parity-1))

def bounder(x):
     return(1 if x >= 0.5 else 0)

#Create Data
lengthTrain = 1000
lengthTest = 1000
parity = 3
input_shift = [0]
(u,y) = generateParity(lengthTrain, parity)

#Create network
esn = ESN(n_inputs = 1, 
          n_outputs = parity - 1,
          n_reservoir = 300,
#          spectral_radius = 0.25,
          sparsity = 0.9,
          noise = 0.01,
#          input_shift = input_shift,
#          input_scaling = [0.01],
#          teacher_scaling = 1.12,
          teacher_shift = -0.7,
          out_activation = np.tanh,
          inverse_out_activation = np.arctanh,
          random_state = rng,
          silent = False)

print('fitting')
pred_train = esn.fit(u, y)

#Assess Training Error     
vec_bounder = np.vectorize(bounder)
bounded = vec_bounder(pred_train)

#print(bounded)
#print('\n', y.reshape(lengthTrain, parity-1))
boundedErrors = np.abs(bounded - y.reshape(lengthTrain, parity-1))
numWrong = sum([1 if xi != 0 else 0 for xi in np.sum(boundedErrors, axis = 1)])
print('Number Wrong (Training):', numWrong, 'Out of', lengthTrain, '\n')


#Testing
(uTest,yTest) = generateParity(lengthTest, parity)
pred_test = esn.predict(uTest)
print('Testing MSE:',np.sqrt(np.mean((pred_test - yTest)**2)))

boundedTest = vec_bounder(pred_test)
boundedTestErrors = np.abs(boundedTest - yTest.reshape(lengthTest, parity-1))
numWrongTest = sum([1 if xi != 0 else 0 for xi in np.sum(boundedTestErrors, axis = 1)])

#print(boundedTest)
#print('\n', yTest.reshape(lengthTest, parity-1))

print('Number Wrong (Testing):', numWrongTest, 'Out of', lengthTest, '\n')
