import tensorflow as tf

class tfESN(tf.keras.layers.Layer):
    
  def __init__(self, n_reservoir):
    super(tfESN, self).__init__(trainable = False)    
    self.n_reservoir = n_reservoir

  def build(self, inputs):
    
    self.n_inputs = tf.shape(inputs)[-1]
     
    #Initialize the reservoir to reservoir weights
    self.W_res = tf.random_normal([self.n_reservoir, self.n_reservoir], mean = 0, stddev = 1, dtype = tf.float32, name = 'WeightsFromReservoir')

    #Initialize the input to reservoir weights
    self.W_in = tf.random_normal([self.n_reservoir, self.n_inputs], mean = 0, stddev = 1, dtype = tf.float32, name = 'WeightsFromInput')
    self.B_in = tf.random_normal([self.n_reservoir,1], mean = 0, stddev = 1, dtype = tf.float32, name = 'Bias')

    #Initialize the output to reservoir weights
    self.W_out = tf.random_normal([self.n_reservoir, 1], mean = 0, stddev = 1, dtype = tf.float32, name = 'WeightsFromOutput')

    #Initialize the current state
    self.x = tf.zeros([self.n_reservoir, 1], name = 'States')
        

  def call(self, inputs):
    partInput = tf.reshape(tf.tensordot(self.W_in, inputs, 1),[self.n_inputs, self.n_reservoir])
    partReservoir = tf.transpose(tf.tensordot(self.W_res, self.x, 1))
    partBias = tf.transpose(self.B_in)
    
    currentState = (partInput + partReservoir + partBias)
    
    return currentState

