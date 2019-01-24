# TFESN2.py
# Parker Kain 
# This code is a framework for testing sequential parity on various neural net architectures.
# Here, we use a reservoir layer created by RCLayer.py followed by a dense layer.
# Changing parity length will allow the user to see how the framework scales to more complicated tasks.



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
    y = np.array(y).reshape(length, parity-1)
    y_lag = np.append(np.array([0]), y)[:-1].reshape(length, parity-1)
    return((u, y, y_lag))

#---------------------------------------------------------------------------
#Hyperparameters 
#---------------------------------------------------------------------------
#tf.set_random_seed(1234)
lengthTrain = 5000
lengthTest = 5000
parity = 2
teacher_shift = -0.7
teacher_scaling = 1.12
n_inputs = 1
n_outputs = parity - 1
n_reservoir = 50
output_activation = tf.tanh
train_file_location = "C:/Users/parke/Documents/Capstone/PythonCode/training"
test_file_location = "C:/Users/parke/Documents/Capstone/PythonCode/testing"

#---------------------------------------------------------------------------
#Define the Data
#---------------------------------------------------------------------------

(u_train, y_train, y_lag_train) = generateParity(lengthTrain, parity) 
(u_test, y_test, y_lag_test) = generateParity(lengthTest, parity) 


#-------------------------------------
#Generate Network
#-------------------------------------

#Make Iterator
u, y, y_lag = tf.placeholder(tf.float32, shape=[None,1]), tf.placeholder(tf.float32, shape=[None,n_outputs]), tf.placeholder(tf.float32, shape=[None,n_outputs])
dataset = tf.data.Dataset.from_tensor_slices(({'u': u, 
                                               'y': y,
                                               'y_lag':y_lag}))    
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

#Make Reservoir
my_ESN = tfESN(n_reservoir, teacher_shift, teacher_scaling)
currentState = my_ESN((next_row['u'], next_row['y_lag']))       

#Readout
readout = tf.layers.Dense(units = n_outputs, 
                          use_bias = True,
                          activation = output_activation,
                          name = 'Readout')

#Get prediction
y_pred = tf.reshape(readout(currentState),[n_outputs,])

#-------------------------------------
#Loss
#-------------------------------------
with tf.name_scope('Training_Parameters'):
    loss = tf.losses.mean_squared_error(labels=next_row['y'], predictions=y_pred)
    #-------------------------------------
    #Training Measurements
    #-------------------------------------
    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(loss)
    acc, acc_op = tf.metrics.accuracy(tf.round(next_row['y']), tf.round(y_pred), name = 'accuracy')
    
#-------------------------------------
#Initialize Variables
#-------------------------------------
init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
sess = tf.Session()
sess.run(init_g)
sess.run(init_l)
sess.run(iterator.initializer, feed_dict={u: u_train, y: y_train, y_lag:y_lag_train})

#-------------------------------------
#Initialize Tensorboard Grpah and Summarise
#-------------------------------------
tf.summary.scalar('loss',loss)
tf.summary.scalar('accuracy', acc)
tf.summary.scalar('other_accuracy', acc_op)

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter(train_file_location)
test_writer = tf.summary.FileWriter(test_file_location)
writer.add_graph(sess.graph)

#sess = tf_debug.LocalCLIDebugWrapperSession(sess) #Debugging

#-------------------------------------
#Training
#-------------------------------------
print('Training ...')
for i in range(lengthTrain):
    #Run one timestep
    (summary, _, current_loss, cur_acc_op, cur_acc) = sess.run((merged_summary, train, loss, acc_op, acc))
    
    #Give status update
    if (i+1) % 1000 == 0:
        print(round(i / lengthTrain * 100), '% complete')  
    
    #Write summary to tensorboard
    if i % 100 == 0:
        writer.add_summary(summary, i)

#-------------------------------------
#Testing
#-------------------------------------
#sess = tf_debug.LocalCLIDebugWrapperSession(sess) #Debugging
print('Switching to Test ... ')

#Reinitialize data with test set
sess.run(iterator.initializer, feed_dict={u: u_test, y: y_test, y_lag:y_lag_test})

for i in range(lengthTest):
    #Do a single passthrough
    (summary, _, current_loss, cur_acc_op, cur_acc) = sess.run((merged_summary, train, loss, acc_op, acc))
    
    #Status update
    if (i+1) % 1000 == 0:
        print(round(i / lengthTest * 100), '% complete') 
    
    #Write to tensorboard
    if i % 100 == 0:
        test_writer.add_summary(summary, i)
