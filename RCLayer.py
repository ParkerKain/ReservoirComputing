import tensorflow as tf

class tfESN(tf.keras.layers.Layer):

  def __init__(self, n_reservoir):
    """Builds any parameters that are not based on the inputs recieved by the layer
    
    args:
        n_reservoir: number of reservoir neurons
    """
    super(tfESN, self).__init__(trainable = False)    
    self.n_reservoir = n_reservoir

  def build(self, inputs):
    """
    This is run when the first input is passed to the layer, and finalized initalization.
    Hence all of the input based weights are defined here.
    
    args:
        inputs: The inputs to pass through the layer, typically a tensor
    
    """
    self.n_inputs = inputs[0] 
    
    #Initialize the reservoir to reservoir weights
    self.W_res = tf.get_variable("tf_esn/WeightsFromReservoir", [self.n_reservoir, self.n_reservoir], trainable = False) * 2 - 1

    #Initialize the input to reservoir weights
    self.W_in = tf.get_variable("tf_esn/WeightsFromInput", [self.n_reservoir, self.n_inputs], trainable = False) * 2 - 1
    self.B_in = tf.get_variable("tf_esn/Bias", [self.n_reservoir, 1], trainable = False) * 2 - 1

    #Initialize the output to reservoir weights
    self.W_out = tf.get_variable("tf_esn/WeightsFromOutput", [self.n_reservoir, 1], trainable = False) * 2 - 1

    #Initialize the current state
    #self.x = tf.zeros([self.n_reservoir, 1], name = 'tf_esn/States')
    self.x = tf.get_variable(name="tf_esn/States", shape=[self.n_reservoir, 1], initializer=tf.zeros_initializer())

  def call(self, inputs):
    """
    This handles the actual passing through of the layer
    
    args:
        inputs: the input to pass through the layer
    """
    partInput = tf.reshape(tf.tensordot(self.W_in, inputs, 1),[self.n_inputs, self.n_reservoir], name = 'partInput')
    partReservoir = tf.transpose(tf.tensordot(self.W_res, self.x, 1), name = 'partReservoir')
    partBias = tf.transpose(self.B_in, name = 'partBias')
    
    currentState = tf.add_n((partInput, partReservoir, partBias), 'CurrentState')
    self.x = currentState

    tf.summary.histogram("partInput", partInput)
    tf.summary.histogram("partReservoir", partReservoir)
    tf.summary.histogram("partBias", partBias)    
    tf.summary.histogram("x", self.x)
    tf.summary.histogram("currentState", currentState)
    tf.summary.histogram("WeightsFromInput", self.W_in)
    tf.summary.histogram("BiasFromInput", self.B_in)
    tf.summary.histogram("WeightsFromOutput", self.W_out)
    tf.summary.histogram("WeightsFromReservoir", self.W_res)

    
    return currentState

