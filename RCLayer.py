import tensorflow as tf

class tfESN(tf.keras.layers.Layer):

  def __init__(self, n_reservoir, teacher_shift, teacher_scale):
    """
    Builds any parameters that are not based on the inputs recieved by the layer
    
    args:
        n_reservoir: number of reservoir neurons
    """
    super(tfESN, self).__init__(trainable = False)    
    self.n_reservoir = n_reservoir
    self.teacher_shift = teacher_shift
    self.teacher_scale = teacher_scale

  def build(self, inputs):
    """
    This is run when the first input is passed to the layer, and finalized initalization.
    Hence all of the input based weights are defined here.
    
    args:
        inputs: The inputs to pass through the layer, typically a tensor
    """
    
    passed = inputs[0]
    
    self.spectral_radius = 0.95
    self.n_inputs = passed[0] 
    
    #Initialize the reservoir to reservoir weights
    self.W_res = tf.get_variable("tf_esn/WeightsFromReservoir", [self.n_reservoir, self.n_reservoir], 
                                 initializer = tf.initializers.random_uniform(minval = -0.5, maxval=0.5),trainable = False)
    
    # compute the spectral radius of these weights:
    (e,v) = tf.linalg.eigh(self.W_res)
    radius = tf.reduce_max(tf.abs(e))
    # rescale them to reach the requested spectral radius:
    self.W_res = self.W_res * (self.spectral_radius / radius)

    #Initialize the input to reservoir weights
    self.W_in = tf.get_variable("tf_esn/WeightsFromInput", [self.n_reservoir, self.n_inputs], 
                                initializer = tf.initializers.random_uniform(minval = -1, maxval=1), trainable = False)
    self.B_in = tf.get_variable("tf_esn/Bias", [self.n_reservoir, 1], trainable = False)

    #Initialize the output to reservoir weights
    self.W_out = tf.get_variable("tf_esn/WeightsFromOutput", [self.n_reservoir, 1], 
                                 initializer = tf.initializers.random_uniform(minval = -1, maxval=1), trainable = False)

    #Initialize the current state
    #self.x = tf.zeros([self.n_reservoir, 1], name = 'tf_esn/States')
    self.x = tf.get_variable(name="tf_esn/States", shape=[self.n_reservoir, 1], initializer=tf.zeros_initializer())

  def call(self, inputs):
    """
    This handles the actual passing through of the layer
    
    args:
        inputs: the input to pass through the layer
    """
    
    passed = inputs[0]
    lastOutput = inputs[1] * self.teacher_scale + self.teacher_shift
    
    partInput = tf.reshape(tf.tensordot(self.W_in, passed, 1),[self.n_inputs, self.n_reservoir], name = 'partInput')
    partReservoir = tf.transpose(tf.tensordot(self.W_res, self.x, 1), name = 'partReservoir')
    partOutput = tf.reshape(tf.tensordot(self.W_out, lastOutput, 1),[self.n_inputs, self.n_reservoir], name = 'partOutput')
    partBias = tf.transpose(self.B_in, name = 'partBias')
    
    
    currentState = tf.tanh(tf.add_n((partInput, partReservoir, partOutput, partBias)), 'CurrentState')
    self.x = currentState

    tf.summary.histogram('passed', passed)
    tf.summary.histogram("partInput", partInput)
    tf.summary.histogram("partReservoir", partReservoir)
    tf.summary.histogram("partOutput", partOutput)
    tf.summary.histogram("partBias", partBias)    
    tf.summary.histogram("x", self.x)
    tf.summary.histogram("currentState", currentState)
    tf.summary.histogram("WeightsFromInput", self.W_in)
    tf.summary.histogram("BiasFromInput", self.B_in)
    tf.summary.histogram("WeightsFromOutput", self.W_out)
    tf.summary.histogram("WeightsFromReservoir", self.W_res)
    
    return currentState