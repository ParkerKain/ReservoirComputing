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


# =============================================================================
# def generateParity(length, parity):
#     """Generates a (length) sized bitstring of (parity) parity 
#     
#     Args:
#         length: the length of the bitrstring to return
#         parity: parity to measure
#     Returns:
#         u: a bitstring of (length) size, randomly generated
#         y: a length_by_(parity)-1 numpy array of parity results. 
#         y_lag : same as above, but offput by one timestep
#     """
#     u = np.random.randint(2, size=length).reshape(length,1)
#     y = []
#     currentParity = 0 
#     
#     for i in range(length):
#         parityState = np.zeros(parity)   
#     
#         currentU = u[i]
#         currentParity = (currentParity + currentU) % parity
#         parityState = np.zeros((1, parity)) 
#         parityState[:,currentParity] = 1
#     
#         y.append(parityState)
#     y = np.array(y).reshape(length, parity)
#     y_lag = np.append(np.array([0]), y)[:-1].reshape(length, parity)
#     return((u, y, y_lag))


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

    print('totalU Original', totalU.shape)

    totalU = np.transpose(totalU, (1,0))
    print('totalU tranposed', totalU.shape)
    
    y = []
    #totalY = np.zeros([length, parity, numStrings])
    totalY = []
    currentParity = 0 
    
    for string in range(numStrings):
        #print(string)
        for i in range(length):
            #print('\ni =', i)
            parityState = np.zeros(parity)   
            
            currentU = totalU[i, string]
            
            #print('currentU:', currentU)
            currentParity = (currentParity + currentU) % parity
            
            #print('currentParity', currentParity)
            
            parityState = np.zeros((1, parity)) 
            parityState[:,currentParity] = 1

            #print('parityState:', parityState)
            
            
            parityState = parityState
            y.append(parityState)
        #y = np.array(y).reshape(length, parity)
        y = np.squeeze(np.array(y))
        #totalY[:,:,string] = y
        totalY.append(y)
        y = []
        currentParity = 0
        
    print('totalY', np.array(totalY).shape)
    print('totalY reshaped', np.transpose(np.array(totalY), (1,2,0)).shape)
    totalY = np.transpose(np.array(totalY), (1,2,0))
    
    print('y_lag test!', (np.append(np.zeros((1,parity,numStrings)), totalY, axis = 0)[:-1]).shape)
    y_lag = np.append(np.zeros((1,parity,numStrings)), totalY, axis = 0)[:-1]
    #y_lag = np.append(np.array([0]), totalY)[:-1].reshape(length, parity, numStrings)
    print('y_lag', y_lag.shape)
    #totalY = np.array(totalY).reshape(length, parity, numStrings)
    return((totalU, totalY, y_lag))


# =============================================================================
#---------------------------------------------------------------------------
#Hyperparameters 
#---------------------------------------------------------------------------
    
#tf.set_random_seed(1234)
readoutLayer = "Dense"
batch_size = 1
numStrings = 1
lengthTrain = 10000
lengthTest = 1000
parity = 20
teacher_shift = -0.7
teacher_scaling = 1.12
n_inputs = 1
n_outputs = parity
n_reservoir = 20
prev_state_weight = 0.1
output_activation = tf.tanh
train_file_location = "C:/Users/kainp1/Documents/GitHub/ReservoirComputing/training"
test_file_location = "C:/Users/kainp1/Documents/GitHub/ReservoirComputing/testing"

#---------------------------------------------------------------------------
#Define the Data
#---------------------------------------------------------------------------
start_time = time.time()
(u_train, y_train, y_lag_train) = generateParity(lengthTrain, parity, numStrings)
print('parity:', parity)
print('batch_size:', batch_size) 
print('Input shape', u_train.shape)
print('Output shape', y_train.shape)

#(u_test, y_test, y_lag_test) = generateParity(lengthTest, parity, numStrings) 

#-------------------------------------
#Generate Network
#-------------------------------------

#Make Iterator
u, y, y_lag = (tf.placeholder(tf.float32, shape=[None,batch_size]), 
              tf.placeholder(tf.float32, shape=[None,n_outputs,batch_size]), 
              tf.placeholder(tf.float32, shape=[None,n_outputs,batch_size]))

dataset = tf.data.Dataset.from_tensor_slices(({'u': u, 
                                               'y': y,
                                               'y_lag':y_lag}))
    
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

print('next_row["u"] from iterator:', next_row['u'].shape)
print('next_row["y"] from iterator:', next_row['y'].shape)

#Make Reservoir
my_ESN = tfESN(n_reservoir, batch_size, n_inputs, teacher_shift, teacher_scaling, prev_state_weight)

print('---------------------------------------------------------')
print('Entering Reservoir')
currentState = my_ESN((next_row['u'], next_row['y_lag']))       
print('---------------------------------------------------------')
print('Exiting Reservoir')
combinedStates = tf.concat(currentState, axis = 0)
#Readout initializing
if readoutLayer == "Dense":
    
    print('combinedStates', combinedStates.shape)

    readout = tf.layers.Dense(units = n_outputs, 
                              use_bias = True,
                              activation = output_activation,
                             name = 'ReadoutDense')
    y_preds = readout(combinedStates)
    print('Readout', y_preds)
    
    #y_preds = []
    #for i in range(batch_size):
    #    y_preds.append(tf.reshape(readout(currentState[i]),[parity,1]))

elif readoutLayer == "GRU":
    readout = tf.keras.layers.GRU(units = n_outputs,
                                  name = 'ReadoutGRU')
    
    Transformed_3D = readout(tf.reshape(currentState, [1,n_reservoir,1]))
    y_pred = tf.reshape(Transformed_3D,[n_outputs,batch_size])

#-------------------------------------
#Loss
#-------------------------------------
with tf.name_scope('Training_Parameters'):
    loss = tf.losses.mean_squared_error(labels=next_row['y'], predictions=tf.reshape(y_preds, [parity, batch_size]))
    #loss = tf.losses.mean_squared_error(labels=next_row['y'], predictions=tf.transpose(y_preds, (1,0)))

    #print(loss)
    #-------------------------------------
    #Training Measurements
    #-------------------------------------
    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(loss)
    acc, acc_op = tf.metrics.accuracy(tf.round(next_row['y']), tf.round(tf.reshape(y_preds, [parity, batch_size])), name = 'accuracy')
    #acc, acc_op = tf.metrics.accuracy(tf.round(tf.reshape(next_row['y'][:,0],[2,1])), tf.round(y_pred), name = 'accuracy')
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
preds = []
print('Training ...')
for i in range(lengthTrain):
    #Run one timestep
    (summary, _, current_loss, cur_acc_op, cur_acc, predictions) = sess.run((merged_summary, train, loss, acc_op, acc, y_preds))
    preds.append(predictions)
    #print('------------------------------------------')
    #print('prediction shape')
    #print(np.array(predictions).shape)
    #print('overall prediction list shape')
    #print(np.array(preds).shape)
    #print('Before reshaping')
    #print(np.array(predictions))
    #print('After reshaping')
    #print(np.array(predictions).reshape(1, 3, 2))
    
    #Give status update
    if (i+1) % 1000 == 0:
        print(round(i / lengthTrain * 100), '% complete')  
    
    #Write summary to tensorboard
    if (i % 100 == 0) and (i != 0):
        
        #Calculate and write hamming accuracy!
        
        last100Preds = np.transpose(np.array(preds[i-100:i]), (1,2,0))
        print(last100Preds.shape)
        #last100Preds = last100Preds.reshape([100,parity, batch_size])
        oneHotPreds = (last100Preds == last100Preds.max(axis=1)[:,None]).astype(int)
        print(oneHotPreds[:,:,0])
        #print(oneHotPreds.shape)
        last100True = np.transpose(np.array(y_train[i-100:i]), (2,1,0))
        #print(last100True.shape)
        print(last100True[:,:,0])
        
        #print(oneHotPreds)
        #print(last100True[0])
        #print(oneHotPreds[0] - last100True[0])
        
        print('----------------------------------')
        #print(np.abs(oneHotPreds - last100True))
        #print(np.sum(np.abs(oneHotPreds - last100True)))
        hammingAcc = 1 - np.sum(np.abs(oneHotPreds - last100True)) / parity / 100
        print('Accuracy for last 100:', hammingAcc * 100, '%')

        writer.add_summary(summary, i)


#-------------------------------------
#Testing
#-------------------------------------
#sess = tf_debug.LocalCLIDebugWrapperSession(sess) #Debugging
print('Switching to Test ... ')

#Reinitialize data with test set
sess.run(iterator.initializer, feed_dict={u: u_test, y: y_test, y_lag:y_lag_test})

preds = []
for i in range(lengthTest):
    #Do a single passthrough
    (summary, _, current_loss, cur_acc_op, cur_acc, predictions) = sess.run((merged_summary, train, loss, acc_op, acc, y_pred))
    preds.append(predictions)
    
    #Status update
    if (i+1) % 1000 == 0:
        print(round(i / lengthTest * 100), '% complete') 
    
    #Write to tensorboard
    if (i % 100 == 0) and (i != 0):
        
        #Calculate and write hamming accuracy!
        last100Preds = np.array(preds[i-100:i])
        oneHotPreds = (last100Preds == last100Preds.max(axis=1)[:,None]).astype(int)
        last100True = np.array(y_test[i-100:i])
        
        
        hammingAcc = 1 - np.sum(np.abs(oneHotPreds - last100True)) / 2 / 100
        print('Accuracy for last 100:', hammingAcc * 100, '%')

        test_writer.add_summary(summary, i)

print(time.time() - start_time)
