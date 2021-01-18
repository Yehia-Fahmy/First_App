import numpy as np
import keras
from keras import Model as M
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import time as t
import pickle
from keras.utils import to_categorical

# paths to images
p_train_cats = 'cats'
p_train_dogs = 'dogs'
p_test_cats = 'test_cats'
p_test_dogs = 'test_dogs'


# global variables
IMG_SIZE = 240
NUM_EPOCHS = 1
BATCH_SIZE = 10

# function definitions

# function to convert the time into something readable
def convert_time(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)


# function to load .pickle files
def load_data(file_name):
    print(f'Loading {file_name}...')
    file = pickle.load(open(file_name, 'rb'))
    return file


# quick function to show the image
def show(img):
    plt.imshow(img, cmap='gray')
    plt.show()


# reshapes the images to the right size
def reshape_data(X, y):
    print(f"Reshaping data...")
    X = np.array(X)     # ensuring that lists are instead arrays
    training_data = X / 255
    training_data = np.array(training_data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)
    return training_data, y


# function to build the network
def build_network():
    mobile = keras.applications.mobilenet.MobileNet()
    x = mobile.layers[-6].output
    predictions = Dense(2, activation='softmax')(x)
    model = M(inputs=mobile.input, outputs=predictions)
    # makes only the last 5 layers trainable
    for layer in model.layers[:-5]:
        layer.trainable = False
    model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


# function to train the model
def train_model(model, imgs, labels):
    print("Training model...")
    model.fit(imgs, labels, epochs=NUM_EPOCHS, validation_split=0.1, batch_size=BATCH_SIZE)
    return model


# Code to run
start_time = t.time()
print("Starting...")

# load in data
training_images = load_data('Images.pickle')
training_labels = load_data('Labels.pickle')
testing_images = load_data('Testing_Images.pickle')
testing_labels = load_data('Testing_Labels.pickle')

# reshape the data
training_images, training_labels = reshape_data(training_images, training_labels)
testing_images, testing_labels = reshape_data(testing_images, testing_labels)

our_model = build_network()
trained_model = train_model(our_model, training_images, training_labels)

# prints the elapsed time for convenience
total_time = t.time() - start_time
total_time = round(total_time, 2)
total_time = convert_time(total_time)

# final message
print(f"Finished in: {total_time}")
print('Success!')