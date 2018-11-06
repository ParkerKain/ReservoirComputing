import tensorflow as tf

class tfESN(tf.keras.layers.Layer):
    
  def __init__(self, n_reservoir):
    super(tfESN, self).__init__(trainable = False)    
    self.n_reservoir = n_reservoir

  def build(self, inputs):
    
    self.n_inputs = inputs[0] 
    
    #Initialize the reservoir to reservoir weights
    #self.W_res = tf.random_normal([self.n_reservoir, self.n_reservoir], mean = 0, stddev = 1, dtype = tf.float32, name = 'WeightsFromReservoir', seed=1234)
    self.W_res = tf.get_variable("tf_esn/WeightsFromReservoir", [self.n_reservoir, self.n_reservoir], trainable = False)

    #Initialize the input to reservoir weights
    #self.W_in = tf.random_normal([self.n_reservoir, self.n_inputs], mean = 0, stddev = 1, dtype = tf.float32, name = 'WeightsFromInput', seed=5678)
    #self.B_in = tf.random_normal([self.n_reservoir,1], mean = 0, stddev = 1, dtype = tf.float32, name = 'Bias', seed=9012)
    self.W_in = tf.get_variable("tf_esn/WeightsFromInput", [self.n_reservoir, self.n_inputs], trainable = False)
    self.B_in = tf.get_variable("tf_esn/Bias", [self.n_reservoir, 1], trainable = False)

    #Initialize the output to reservoir weights
    #self.W_out = tf.random_normal([self.n_reservoir, 1], mean = 0, stddev = 1, dtype = tf.float32, name = 'WeightsFromOutput', seed=3456)
    self.W_out = tf.get_variable("tf_esn/WeightsFromOutput", [self.n_reservoir, 1], trainable = False)

    #Initialize the current state
    self.x = tf.zeros([self.n_reservoir, 1], name = 'States')
    

  def call(self, inputs):
    partInput = tf.reshape(tf.tensordot(self.W_in, inputs, 1),[self.n_inputs, self.n_reservoir])
    partReservoir = tf.transpose(tf.tensordot(self.W_res, self.x, 1))
    partBias = tf.transpose(self.B_in)
    
    currentState = (partInput + partReservoir + partBias)
    
    return currentState

