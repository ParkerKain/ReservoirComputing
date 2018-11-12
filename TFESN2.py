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

lengthTrain = 1000
lengthTest = 100
parity = 2
n_outputs = parity - 1

(u_train, y_train) = generateParity(lengthTrain, parity) 
(u_test, y_test) = generateParity(lengthTest, parity) 

#-------------------------------------
#Generate Network
#-------------------------------------

#Make Iterator
u, y = tf.placeholder(tf.float32, shape=[None,1]), tf.placeholder(tf.float32, shape=[None,1])
dataset = tf.data.Dataset.from_tensor_slices(({'u': u, 
                                               'y': y}))    
    
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

#Get prediction
y_pred = tf.reshape(readout(currentState),[1,])

#-------------------------------------
#Loss
#-------------------------------------
with tf.name_scope('Training_Parameters'):
    loss = tf.losses.mean_squared_error(labels=next_row['y'], predictions=y_pred)
    #-------------------------------------
    #Training Measurements
    #-------------------------------------
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    
#-------------------------------------
#Initialize Variables
#-------------------------------------
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
sess.run(iterator.initializer, feed_dict={ u: u_train, y: y_train})

#-------------------------------------
#Initialize Tensorboard Grpah and Summarise
#-------------------------------------
tf.summary.scalar('loss',loss)

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter(".")
writer.add_graph(sess.graph)

#sess = tf_debug.LocalCLIDebugWrapperSession(sess) #Debugging

#-------------------------------------
#Training
#-------------------------------------
print('Training ...')
outputsPred = []
outputsTrain = []
for i in range(lengthTrain):
    (summary, _, current_loss, predicted, actual) = sess.run((merged_summary, train, loss, y_pred, next_row['y']))
    outputsPred.append(predicted)
    outputsTrain.append(actual)
    if i >= 100:
        accuracy = np.sum(np.abs(np.round(np.array(outputsPred) - np.array(outputsTrain)))/ 100)
        tf.summary.scalar('accuracy',accuracy)
        outputsPred = outputsPred[1:101]
        outputsTrain = outputsTrain[1:101]
    if (i+1) % 1000 == 0:
        print(round(i / lengthTrain * 100), '% complete')        
    writer.add_summary(summary, i)
    #print(current_loss)

#-------------------------------------
#Testing
#-------------------------------------
print('Switching to Test ... ')
sess.run(iterator.initializer, feed_dict={u: u_test, y: y_test})
for i in range(lengthTest):
    test_predicted = sess.run((y_pred))
    #print(test_predicted)
