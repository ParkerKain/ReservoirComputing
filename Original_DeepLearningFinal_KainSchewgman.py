from keras.models import Sequential
from keras.layers import Dense, GRU
import tensorflow as tf
from keras import backend as K
import numpy as np


def build_model(numGRUneurons):
    """ Creates a simple GRU network, with one layer of GRU neurons, dropout, and a single dense layer.
        Loss is binary crossentropy, optimizer is adam.""" 
    model = Sequential()
    model.add(GRU(numGRUneurons, input_shape=(None, 1), return_sequences=True))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['binary_accuracy'])
    return model

# generating random data
def generate_data(num_points, seq_length):
    """ Generates num_points number of bitstrings of length seq_length, as well as a parity bit.""" 
    x = np.random.randint(2, size=(num_points, seq_length, 1))
    y = x.cumsum(axis=1) % 2
    return x, y

def main():
    """Build and train a simple GRU network for learning sequential parity"""
    #Clearing Keras and Tensorflow Sessions
    K.clear_session()
    tf.reset_default_graph()

    #Specify hyperparameters
    maxEpochs = 30
    batch_size = 200
    bitstringLength = 50
    numGRUneurons = 5
    
    #Generate a training set of 100,000 bitstrings of a given length
    X_train, y_train = generate_data(100000, bitstringLength)
    #Also generates a test set of 1000 bitstrings of length 1,000.
    X_test, y_test = generate_data(1000, 1000)
    
    model = build_model(numGRUneurons)
    
    #Begin Learning Process
    print('Beginning Training...')
    for e in range(maxEpochs):
        print('\nCurrent Epoch:', e+1)
        model.fit(
            X_train, y_train,
            batch_size=batch_size)
        
        score = model.evaluate(X_test, y_test,
                               batch_size=batch_size,
                               verbose=0)
        print('Current Test Set Accuracy:', score[1])
        if score[1] == 1:
            print('Test set accuracy is 100%. Stopping...')
            break

    print('\nFinal Test Accuracy:', score[1])
    print('Final Test Loss:', score[0])

    #Print a few examples.
    print('\n\nPrinting a few examples:')
    num_examples = 5
    X_example, y_example = generate_data(num_examples, bitstringLength)
    yStar = model.predict(X_example, batch_size = None, verbose = 0, steps = 1)
    for i in range(num_examples):
       print('\n-----------------------------------------------------------------------------------')
       print('Trial Input Number', i + 1)
       print('\nX input:', X_example[i,:,:].reshape(bitstringLength))
       print('\nActual Y output:', y_example[i,:,:].reshape(bitstringLength))
       yStarRounded = np.round(yStar[i].reshape(bitstringLength), decimals = 0) #Reshaped for display purposes.
       print('\nPredicted Y output:', yStarRounded)
       yRounded = y_example[i,:,:].reshape(bitstringLength)
       print('\nIncorrect Guesses =', np.sum(yStarRounded - yRounded))
    
#-----------------------------------------------------------------------------------------
main()
    