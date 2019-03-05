# TFESN2.py
# Parker Kain 
# This code is a framework for testing sequential parity on various neural net architectures.
# Here, we use a reservoir layer created by RCLayer.py followed by a dense layer.
# Changing parity length will allow the user to see how the framework scales to more complicated tasks.
import time 
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import numpy as np
from RCLayer import tfESN
from tensorflow.python import debug as tf_debug


tf.reset_default_graph()

#----------------------------------------------------------------------

def generateParity(length, parity, numStrings = 1):
    """Generates a (length) sized bitstring of (parity) parity 
    
    Args:
        length: the length of the bitrstring to return
        parity: parity to measure
        numStrings: number of bitstrings to create
    Returns:
        u: a bitstring of (length) size, randomly generated
        y: a length_by_(parity)-1_by_numStrings numpy array of parity results. 
        y_lag : same as above, but offput by one timestep
    """
    totalU = []
    
    for i in range(numStrings):
        u = np.random.randint(2, size=length)
        totalU.append(u)
        
    totalU = (np.array(totalU))
    totalU = np.transpose(totalU, (1,0))    
    y = []
    totalY = []
    currentParity = 0 
    
    for string in range(numStrings):
        for i in range(length):
            parityState = np.zeros(parity)   
            
            currentU = totalU[i, string]
            currentParity = (currentParity + currentU) % parity
            
            parityState = np.zeros((1, parity))
            parityState[:,currentParity] = 1
            
            parityState = parityState
            y.append(parityState)

        y = np.squeeze(np.array(y))
        totalY.append(y)
        y = []
        currentParity = 0
        
    totalY = np.transpose(np.array(totalY), (1,2,0))
    
    y_lag = np.append(np.zeros((1,parity,numStrings)), totalY, axis = 0)[:-1]
    return((totalU, totalY, y_lag))


# =============================================================================
#---------------------------------------------------------------------------
#Hyperparameters 
#---------------------------------------------------------------------------
    
#tf.set_random_seed(1234)
problem = 'Parity'
readoutLayer = "Dense"
batch_size = 10 #MUST BE 40 FOR PARKINSONS
numStrings = 10 #MUST BE 40 FOR PARKINSONS
lengthTrain = 5000
lengthTest = 1000
parity = 2 #REALLY Is the number of outputs, must be 2 for Parkinsons
teacher_shift = -0.7
teacher_scaling = 1.12
n_inputs = 1 #BE CAREFUL, must be 5 for parkinsons ATM
n_outputs = parity
n_reservoir = 100
prev_state_weight = 0.1
output_activation = tf.tanh
train_file_location = "C:/Users/kainp1/Documents/GitHub/ReservoirComputing/training"
test_file_location = "C:/Users/kainp1/Documents/GitHub/ReservoirComputing/testing"

#---------------------------------------------------------------------------
#Define the Data
#---------------------------------------------------------------------------
start_time = time.time()
if problem == 'Parity':
    (u_train, y_train, y_lag_train) = generateParity(lengthTrain, parity, numStrings)
    u_train = np.expand_dims(u_train, 1)
    (u_test, y_test, y_lag_test) = generateParity(lengthTest, parity, numStrings) 
    y_lag_test = np.zeros((lengthTest, parity, batch_size))
    u_test = np.expand_dims(u_test, 1)

    
if problem == 'Parkinsons':
    
    u_train = np.load('u_train.npy')
    y_train = np.load('y_train.npy')
    y_lag_train = np.load('y_lag_train.npy')

    print(u_train.shape)
    print(y_train.shape)
    print(y_lag_train.shape)
    
    print('NOT YET IMPLEMENTED')
#-------------------------------------
#Generate Network
#-------------------------------------
#Make Iterator
u, y, y_lag = (tf.placeholder(tf.float32, shape=[None, None, batch_size]), 
              tf.placeholder(tf.float32, shape=[None, n_outputs, batch_size]), 
              tf.placeholder(tf.float32, shape=[None, n_outputs, batch_size]))

predictions = np.zeros((parity, batch_size))
y_preds_copy = tf.placeholder(tf.float32, shape=[n_outputs, batch_size])
prob = tf.placeholder(tf.float32, shape = [None])

probability = np.linspace(start= 50, stop = 0, num = lengthTrain)
print(probability.shape)

dataset = tf.data.Dataset.from_tensor_slices(({'u': u, 
                                               'y': y,
                                               'y_lag':y_lag,
                                               'prob':prob}))
    
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

#Make Reservoir
my_ESN = tfESN(n_reservoir, batch_size, n_inputs,lengthTrain,  teacher_shift, teacher_scaling, prev_state_weight)
#my_ESN2 = tfESN(n_reservoir, batch_size, n_inputs, teacher_shift, teacher_scaling, prev_state_weight)

currentState = my_ESN((next_row['u'], next_row['y_lag'], y_preds_copy, next_row['prob']))    
combinedStates = tf.concat(currentState, axis = 0)

#Readout initializing
if readoutLayer == "Dense":
    
    readout = tf.layers.Dense(units = n_outputs, 
                              use_bias = True,
                              activation = output_activation,
                             name = 'ReadoutDense')
    y_preds = readout(combinedStates)
    #y_preds_copy = tf.assign(y_preds_copy, y_preds)

elif readoutLayer == "GRU":
    readout = tf.keras.layers.GRU(units = n_outputs,
                                  name = 'ReadoutGRU')

    Transformed_3D = readout(tf.reshape(combinedStates, [batch_size,n_reservoir,1]))
    y_preds = Transformed_3D

#-------------------------------------
#Loss
#-------------------------------------
with tf.name_scope('Training_Parameters'):
    loss = tf.losses.mean_squared_error(labels=next_row['y'], predictions=tf.transpose(y_preds, (1,0)))

    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(loss)
    acc, acc_op = tf.metrics.accuracy(tf.round(next_row['y']), tf.round(tf.reshape(y_preds, [parity, batch_size])), name = 'accuracy')
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope = "accuracy")
    
#-------------------------------------
#Initialize Variables
#-------------------------------------
init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
config = tf.ConfigProto(allow_soft_placement = True)
sess = tf.Session(config = config)
sess.run(init_g)
sess.run(init_l)
sess.run(iterator.initializer, feed_dict={u: u_train, y: y_train, y_lag:y_lag_train, prob:probability})

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
preds = []
print('Training ...')
for i in range(lengthTrain):
    #Run one timestep
    (summary, _, current_loss, cur_acc_op, cur_acc, predictions) = sess.run((merged_summary, train, loss, acc_op, acc, y_preds), feed_dict = {y_preds_copy:predictions})
    preds.append(predictions)
    predictions = np.transpose(predictions, (1,0))

    #Give status update
    if (i+1) % 1000 == 0:
        print(round(i / lengthTrain * 100), '% complete')  
    
    #Write summary to tensorboard
    if (i % 100 == 0) and (i != 0):
        
        #Calculate and write hamming accuracy!
        
        last100Preds = np.transpose(np.array(preds[i-100:i]), (1,2,0))
        oneHotPreds = (last100Preds == last100Preds.max(axis=1)[:,None]).astype(int)
        last100True = np.transpose(np.array(y_train[i-100:i]), (2,1,0))
        print('----------------------------------')
        print('Number Incorrect in last 100:', np.sum(np.abs(oneHotPreds - last100True)))
        hammingAcc = 1 - np.sum(np.abs(oneHotPreds - last100True)) / parity / 100 / batch_size
        print('Accuracy for last 100:', hammingAcc * 100, '%')

        writer.add_summary(summary, i)


#-------------------------------------
#Testing
#-------------------------------------
#sess = tf_debug.LocalCLIDebugWrapperSession(sess) #Debugging
print('Switching to Test ... ')

#Reinitialize data with test set
probability = np.linspace(start= 0, stop = 0, num = lengthTest)
sess.run(iterator.initializer, feed_dict={u: u_test, y: y_test, y_lag:y_lag_test, prob:probability})

preds = []
for i in range(lengthTest):
    #Run one timestep
    (summary, _, current_loss, cur_acc_op, cur_acc, predictions) = sess.run((merged_summary, train, loss, acc_op, acc, y_preds), {y_preds_copy:predictions})
    preds.append(predictions)
    predictions = np.transpose(predictions, (1,0))

    
    #Give status update
    if (i+1) % 1000 == 0:
        print(round(i / lengthTest * 100), '% complete')  
    
    #Write summary to tensorboard
    if (i % 100 == 0) and (i != 0):
        
        #Calculate and write hamming accuracy!
        
        last100Preds = np.transpose(np.array(preds[i-100:i]), (1,2,0))
        oneHotPreds = (last100Preds == last100Preds.max(axis=1)[:,None]).astype(int)

        last100True = np.transpose(np.array(y_test[i-100:i]), (2,1,0))
        print('----------------------------------')
        print(np.sum(np.abs(oneHotPreds - last100True)))
        hammingAcc = 1 - np.sum(np.abs(oneHotPreds - last100True)) / parity / 100 / batch_size
        print('Accuracy for last 100:', hammingAcc * 100, '%')

        test_writer.add_summary(summary, i)

print('Runtime:', round(time.time() - start_time,2), 'seconds')
