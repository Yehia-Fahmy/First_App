from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np
import time as t


# function definitions
# function to load the images
def load_data():
    print("Loading data...")
    X = pickle.load(open("Images.pickle", "rb"))
    y = pickle.load(open("Labels.pickle", "rb"))
    return X, y


# reshapes the images to the right size
def reshape_data(X, y):
    print("Reshaping data...")
    X = np.array(X)     # ensuring that lists are instead arrays
    X = X / 255         # normalizing the data
    y = np.array(y)
    return X, y


# function to build the network
def build_network(images):
    print("Building network...")

    model = Sequential()
    for i in range(NUMLAYERS):      # adds a layer
        model.add(Conv2D(NUMNODES, (3, 3), input_shape=images.shape[1:]))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation("sigmoid"))        # the final layer is responsible for the prediction

    print("Compiling model...")
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])     # compiles the model

    return model


# function to train the model
def train_model(model, images, labels):
    print("Training model...")
    trained_model = model.fit(images, labels, epochs=NUMEPOCHS, validation_split=0.1, batch_size= BATCHSIZE)
    return trained_model


# Global Variables
NUMLAYERS = 3
NUMNODES = 124
NUMEPOCHS = 3  # number of epochs we want to train for
BATCHSIZE = 32  # higher batch size will train faster

# Code to run


