from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np
import time as t
import matplotlib.pyplot as plt


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
    training_data = X / 255
    training_data = np.array(training_data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    print(f"X.shape: {X.shape}")
    print(f"training_data.shape: {training_data.shape}")
    y = np.array(y)
    return training_data, y


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
    model.summary()
    print("Finished Compiling!")

    return model


# function to train the model
def train_model(model, images, labels):
    print("Training model...")
    trained_model = model.fit(images, labels, epochs=NUMEPOCHS, validation_split=0.1, batch_size=BATCHSIZE)
    return trained_model


# function to convert the time into something readable
def convert_time(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)


# quick function to show the image
def show(img):
    plt.imshow(img, cmap='gray')
    plt.show()


# Global Variables
NUMLAYERS = 4
NUMNODES = 64
NUMEPOCHS = 50  # number of epochs we want to train for
BATCHSIZE = 40  # higher batch size will train faster
IMG_SIZE = 240  # images will be 240 by 240

# Code to run
start_time = t.time()
print("Starting...")

images, labels = load_data()
images, labels = reshape_data(images, labels)
our_model = build_network(images)
our_model_trained = train_model(our_model, images, labels)

# prints the elapsed time for convenience
total_time = t.time() - start_time
total_time = round(total_time, 2)
total_time = convert_time(total_time)

# final message
print(f"Finished in: {total_time}")
print('Success!')
