import tensorflow as tf
import numpy as np
from RCLayer import tfESN
from tensorflow.python import debug as tf_debug


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
        
    return((u, np.array(y).reshape(length, parity-1)))

#---------------------------------------------------------------------------
#Hyperparameters 
#-------------------------------------
tf.set_random_seed(1234)
n_inputs = 100
n_reservoir = 5
n_readout = 1

#-------------------------------------
#Define the Data
#-------------------------------------

lengthTrain = 100
lengthTest = 1000
parity = 2
n_outputs = parity - 1

(u, y_true) = generateParity(lengthTrain, parity) 

#-------------------------------------
#Generate Network
#-------------------------------------

#Make Iterator
dataset = tf.data.Dataset.from_tensor_slices(
            {'u': tf.convert_to_tensor(u, dtype=tf.float32),
             'y': tf.convert_to_tensor(y_true, dtype=tf.float32)})
    
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

#Make Reservoir
my_ESN = tfESN(10)
currentState = my_ESN(next_row['u'])       

#Readout
readout = tf.layers.Dense(units = 1, 
                          use_bias = True,
                          activation = tf.tanh,
                          name = 'Readout')

y_pred = tf.reshape(readout(currentState),[1,])

#-------------------------------------
#Loss
#-------------------------------------
with tf.name_scope('Training_Parameters'):
    loss = tf.losses.mean_squared_error(labels=next_row['y'], predictions=y_pred)

    #-------------------------------------
    #Training
    #-------------------------------------
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
sess.run(iterator.initializer)

#sess = tf_debug.LocalCLIDebugWrapperSession(sess) #Debugging
for i in range(100):
    _, current_loss = sess.run((train,loss))
    print(current_loss)
    
#-------------------------------------
#Output Graph
#-------------------------------------
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
