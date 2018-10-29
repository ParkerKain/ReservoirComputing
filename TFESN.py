import tensorflow as tf
import numpy as np

tf.reset_default_graph()

def generateParity(length, parity):
    """Generates a (length) sized bitstring of (parity) parity 
    
    Args:
        length: the length of the bitrstring to return
        parity: parity to measure
    Returns:
        u: a bitstring of (length) size, randomly generated
        y: a length_by_(parity)-1 numpy array of parity results. 
    """
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
        
        parityState = np.zeros((1, parity-1)) 
        if currentParity != 0:
            parityState[:,currentParity-1] = 1
        
        parityState = parityState
        y.append(parityState)
        
    return(u, np.array(y).reshape(length, parity-1))

#Define Model

class tfESN():
    """Creates a reservoir with specified hyperparameters, to be connected to some
    inputs through tensorflow
    
    Args:
        n_inputs: Number of inputs to be passed through the reservoir in a given timestep
        n_reservoir: Number of neurons in the reservoir layer
    """
    
    def __init__(self, n_inputs, n_reservoir, n_outputs, n_readout):
        """
        Args: 
            n_inputs: length of the input for a given timestep
            n_reservoir: number of reservoir neurons
            n_outputs: number of final outputs to the model
            n_readout: number of neurons in the readout layer.
        """
        #Keep track of all of our class variables
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_readout = n_readout
        
        #Initialize the input weights
        #self.W_in = tf.random_normal([self.n_reservoir,1], mean=0, stddev=1, name = 'WeightsFromInputs')
        self.W_in = tf.convert_to_tensor(np.random.rand(self.n_reservoir, self.n_inputs) * 2 - 1, dtype = tf.float32, name = 'WeightsFromInput')
        self.B_in = tf.convert_to_tensor(np.random.rand(self.n_reservoir,1) * 2 - 1, dtype = tf.float32, name = 'Bias')

        #Initialize the reservoir to reservoir weights
        self.W_res = tf.convert_to_tensor(np.random.rand(self.n_reservoir, self.n_reservoir) * 2 - 1, dtype = tf.float32, name = 'WeightsFromReservoir')

        #Initialize the output to reservoir weights
        self.W_out = tf.convert_to_tensor(np.random.rand(self.n_reservoir, 1) * 2 - 1, dtype = tf.float32, name = 'WeightsFromOutput')

        #Initialize the current state
        self.x = tf.zeros([self.n_reservoir, 1], name = 'States')
        
        #Create the readout layer
        self.currentState = tf.placeholder(tf.float32, shape=[1, n_reservoir], name = 'State')
        
        self.readout = tf.layers.Dense(units = self.n_readout,
                                       use_bias = True,
                                       name = 'Readout')

        PLZ = self.readout(self.currentState)
    def update(self, u):
        """
        Given an input, passes it through the reservoir and returns the state of the reservoir.
        
        Args:
            u: current input pattern for a given timestep
        """
        u = tf.cast(u, dtype=tf.float32)
        
        #Reservoir
        partInput = tf.reshape(tf.tensordot(self.W_in, u, 1),[20,1])
        partReservoir = tf.tensordot(self.W_res, self.x, 1)
        partBias = self.B_in
        currentState = (partInput + partReservoir + partBias)
                        
        #Readout
        output = self.readout(tf.reshape(currentState, shape = [1, 20]))
        
        return(output)
        
    def fit(self, trainInputs, trainOutputs):
        """
        Collect the network's reaction to training data, train readout weights.
        
        Args:
            trainInputs: (numberOfTimesteps) x (numberOfInputs) numpy array
            trainOutputs: (numberOfTimesteps) x (numberOfOutputs) numpy array
        """
        
        dataset = tf.data.Dataset.from_tensor_slices(u)
        iterator = dataset.make_initializable_iterator()
        next_row = iterator.get_next()
        sess.run(iterator.initializer)

        
        while True:
            try:
               updated = sess.run(self.update(tf.cast(next_row, tf.float32)))
            except tf.errors.OutOfRangeError:
                break
        
        return updated

#Define Data
lengthTrain = 1
lengthTest = 1000
parity = 2
readout_neurons = 1

(u,y) = generateParity(lengthTrain, parity)        

#Initialize Model
n_inputs = 1
n_reservoir = 20
n_outputs = parity - 1

my_res = tfESN(n_inputs = n_inputs, 
               n_reservoir = n_reservoir, 
               n_outputs = n_outputs, 
               n_readout = readout_neurons)

test = my_res.update(u)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#Iterate
updated = my_res.fit(u, y)

print(updated)

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
