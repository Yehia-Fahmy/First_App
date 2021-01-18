import numpy as np
import keras
from keras import Model as M
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
import cv2
import os
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
IMG_SIZE = 224
NUM_EPOCHS = 1
BATCH_SIZE = 10
KERAS_MODEL_NAME = 'Full_Size_Model.h5'
TF_LITE_MODEL_NAME = 'TF_Lite_Model.tflite'

# function definitions


# gets size of file
def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size


# converts bytes for readability
def convert_bytes(size, unit=None):
    if unit == "KB":
        return 'File size: ' + str(round(size / 1024, 3)) + ' Kilobytes'
    elif unit == "MB":
        return 'File size: ' + str(round(size / (1024 * 1024), 3)) + ' Megabytes'
    else:
        return 'File size: ' + str(size) + ' bytes'


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
    X = X / 255
    single_channel = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    triple_channel = []
    for img1 in single_channel:
        img3 = cv2.merge((img1, img1, img1))
        triple_channel.append(img3)
    triple_channel = np.array(triple_channel)
    y = np.array(y)
    y = to_categorical(y)
    print(f"X.shape(): {triple_channel.shape}, y.shape(): {y.shape}")
    return triple_channel, y


# function to build the network
def build_network():
    mobile = keras.applications.mobilenet.MobileNet()
    x2 = mobile.layers[-6].output
    predictions = Dense(2, activation='softmax')(x2)
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


# print the results to the txt file
def print_results():
    model_results = f'''
    #################################################################
    IMG_SIZE = {IMG_SIZE}
    ACCURACY = {acc}%
    TIME = {total_time}
    {full_bytes}
    '''
    file = open('results.txt', 'a')
    file.write(model_results)
    our_model.summary(print_fn=lambda x: file.write(x + '\n'))
    file.close()


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

# build and train the model
our_model = build_network()
trained_model = train_model(our_model, training_images, training_labels)

# save the model
trained_model.save(KERAS_MODEL_NAME)
full_bytes = convert_bytes(get_file_size(KERAS_MODEL_NAME), "MB")

# evaluate the model
loss, acc = trained_model.evaluate(testing_images, testing_labels,
                                   epochs=10, batch_size=BATCH_SIZE, use_multiprocessing='True')
acc = round(acc * 100, 2)

# prints the elapsed time for convenience
total_time = t.time() - start_time
total_time = round(total_time, 2)
total_time = convert_time(total_time)

# prints the results
print_results()

# final message
print(f"Finished in: {total_time}")
print('Success!')