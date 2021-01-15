import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
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

# function definitions

# function to convert the time into something readable
def convert_time(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)


'''# function to prepare the training and testing batches
def prepare_batches(p_cats, p_dogs):
    labels = ['cat', 'dog']
    for label in labels:
        batch = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input)\
            .flow_from_directory(p_cats, target_size=(IMG_SIZE, IMG_SIZE), batch_size=10)'''

# Code to run
start_time = t.time()
print("Starting...")


# prints the elapsed time for convenience
total_time = t.time() - start_time
total_time = round(total_time, 2)
total_time = convert_time(total_time)

# final message
print(f"Finished in: {total_time}")
print('Success!')