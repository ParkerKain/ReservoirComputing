# RCLayer.py
# Parker Kain 
# Extends the layers class from keras to create a reservoir layer. This allows the user to 
# plug this layer into any tensorflow architecture, which can be used for dequential learning
# with minimal computational overhead. 


import tensorflow as tf

class tfESN(tf.keras.layers.Layer):

  def __init__(self, n_reservoir, 
                       batch_size,
                       n_inputs,
                       teacher_shift = 0, 
                       teacher_scale = 1,
                       prev_state_weight = 0):
    """
    Builds any parameters that are not based on the inputs recieved by the layer
    
    args:
        n_reservoir: number of reservoir neurons
        batch_size: number of inputs to pass into the network at a given timestep
        teacher_shift: amount to shift the true outputs that are fed back into the reservoir
        teacher_scale: amount to scale the true outputs that are fed back into the reservoir
        prev_state_weight: amount to weight the preious weight when calculating the current weight. Defaults to 0.
        n_inputs: number of inputs passed at one time 
    """
    super(tfESN, self).__init__(trainable = False)   
    self.batch_size = batch_size
    self.n_reservoir = n_reservoir
    self.teacher_shift = teacher_shift
    self.teacher_scale = teacher_scale
    self.prev_state_weight = prev_state_weight
    self.n_inputs = n_inputs
    
  def build(self, inputs):
    """
    This is run when the first input is passed to the layer, and finalized initalization.
    Hence all of the input based weights are defined here.
    
    args:
        inputs: A list inputs to pass through the layer. Usually a list of tensors, the first element
                being the input to pass through the reservoir, and the second being the previous 
                timestep's true output.
    """
    
    # ------------------------------------------------------
    # Finish Defining the Reservoir
    # -------------------------------------------------------

    self.spectral_radius = 0.95
    self.n_outputs = inputs[1][0]
    self.n_inputs = self.n_inputs 
    
    #Initialize the reservoir to reservoir weights
    self.W_res = tf.get_variable("tf_esn/WeightsFromReservoir", [self.n_reservoir, self.n_reservoir], 
                                 initializer = tf.initializers.random_uniform(minval = -0.5, maxval=0.5),trainable = False)
        
    # compute the spectral radius of these weights:
    (e,v) = tf.linalg.eigh(self.W_res)
    radius = tf.reduce_max(tf.abs(e))
    # rescale them to reach the requested spectral radius:
    self.W_res = self.W_res * (self.spectral_radius / radius)

    print('Reservoir to Reservoir weights:', self.W_res.shape)

    #Initialize the input to reservoir weights
    #self.W_in = tf.get_variable("tf_esn/WeightsFromInput", [self.n_reservoir, self.n_inputs], 
    #                            initializer = tf.initializers.random_uniform(minval = -1, maxval=1), trainable = False)
    self.W_in = tf.get_variable("tf_esn/WeightsFromInput", [self.n_reservoir, self.n_inputs], 
                                initializer = tf.initializers.random_uniform(minval = -1, maxval=1), trainable = False)
    
    print('Input to Reservoir weights:', self.W_in.shape)
    
    self.B_in = tf.get_variable("tf_esn/Bias", [self.n_reservoir, 1], trainable = False)

    print('Bias to Reservoir weights:', self.B_in.shape)

    #Initialize the output to reservoir weights
    self.W_out = tf.get_variable("tf_esn/WeightsFromOutput", [self.n_reservoir, self.n_outputs], 
                                 initializer = tf.initializers.random_uniform(minval = -1, maxval=1), trainable = False)

    print('Output to Reservoir weights:', self.W_out.shape)

    #Initialize the current state
    self.x = [tf.get_variable(name = ('state' + str(i)),shape=[self.n_reservoir, 1], initializer=tf.zeros_initializer()) for i in range(self.batch_size)]
    #self.x = tf.get_variable(name="tf_esn/States", shape=[self.n_reservoir, 1], initializer=tf.zeros_initializer())

    print('States:', self.x)

    #Initialize a variable to store previous state
    self.prev_state = [tf.reshape(self.x[i], [1, self.n_reservoir]) for i in range(self.batch_size)]
    #self.prev_state = tf.reshape(self.x, [1, self.n_reservoir])

  def call(self, inputs):
    """
    This handles the actual passing through of the layer
    
    args:
        inputs: Same as above. A list inputs to pass through the layer. Usually a list of tensors, 
                the first element being the input to pass through the reservoir, and the second 
                being the previous timestep's true output.
    """
    #Set up passed in inputs to be correct shapes
    passed = inputs[0]
    print('---------------------------------------------------------')
    print('Passed before reshaping', passed.shape)
    passed = tf.reshape(passed, [self.batch_size,self.n_inputs])
    print('Passed After Reshaping', passed.shape)

    #Scale previous output
    lastOutput = inputs[1] * self.teacher_scale + self.teacher_shift
    #Set up variables that will keep hold of states and outputs as they accrue below
    currentStates = []
    
    print('uPassed', inputs[0])
    print('yLag', lastOutput)

    
    #Loop for batches, grab each input, pass it through the reservoir, append outputs.
    for i in range(self.batch_size):
        currentPassed = passed[i]
        currentPrevious = lastOutput[:,i]
        print('---------------------------------------------------------')
        print('currentPassed',currentPassed.shape)
        print('currentPrevious',currentPrevious.shape)
        print('state:', self.x)
        
        partInput = tf.reshape(tf.tensordot(self.W_in, currentPassed, 1),[1, self.n_reservoir], name = 'partInput')
        print('partInput', partInput.shape)
        partReservoir = tf.transpose(tf.tensordot(self.W_res, self.x[i], 1), name = 'partReservoir')
        print('partReservoir', partReservoir.shape)
        partOutput = tf.reshape(tf.tensordot(self.W_out, currentPrevious, 1),[1, self.n_reservoir], name = 'partOutput')
        print('partOutput', partOutput.shape)
        partBias = tf.transpose(self.B_in, name = 'partBias')
        print('partBias', partBias.shape)
        partPrevious = (tf.transpose(self.x[i]))
        print('partPrevious', partPrevious.shape)
        
        currentState = (self.prev_state_weight * partPrevious) + ((1-self.prev_state_weight)*tf.tanh(tf.add_n((partInput, partReservoir, partOutput, partBias)), 'CurrentState'))
        print('currentState', currentState.shape)
        self.x[i] = currentState
        currentStates.append(currentState)
        print('List of States:',currentStates)
        
        tf.summary.histogram('passed', passed)
        tf.summary.histogram("partInput", partInput)
        tf.summary.histogram("partReservoir", partReservoir)
        tf.summary.histogram("partOutput", partOutput)
        tf.summary.histogram("partBias", partBias)    
        #tf.summary.histogram("x", self.x)
        tf.summary.histogram("currentState", currentState)
        tf.summary.histogram("WeightsFromInput", self.W_in)
        tf.summary.histogram("BiasFromInput", self.B_in)
        tf.summary.histogram("WeightsFromOutput", self.W_out)
        tf.summary.histogram("WeightsFromReservoir", self.W_res)
        
    #return currentState
    return currentStates

