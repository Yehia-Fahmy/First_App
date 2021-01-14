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


# loads testing data
def load_testing_data():
    print("Loading data...")
    X = pickle.load(open("Testing_Images.pickle", "rb"))
    y = pickle.load(open("Testing_Labels.pickle", "rb"))
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

    model.add(Conv2D(NUMNODES, (3, 3), input_shape=images.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(NUMNODES, (1, 1)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    temp = int(NUMNODES/2)
    model.add(Conv2D(temp, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(temp, (1, 1)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    '''for i in range(NUMLAYERS):      # adds a layer
        model.add(Conv2D(NUMNODES, (3, 3), input_shape=images.shape[1:]))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))'''

    model.add(Flatten())
    model.add(Dense(2))
    model.add(Activation("softmax"))        # the final layer is responsible for the prediction

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
    model.fit(images, labels, epochs=NUMEPOCHS, validation_split=0.1, batch_size=BATCHSIZE)
    return model


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
NUMLAYERS = 2
NUMNODES = 250
NUMEPOCHS = 8  # number of epochs we want to train for
BATCHSIZE = 15  # higher batch size will train faster
IMG_SIZE = 120  # images will be 120 by 120

# Code to run
start_time = t.time()
print("Starting...")

images, labels = load_data()
testing_images, testing_labels = load_testing_data()

images, labels = reshape_data(images, labels)
testing_images, testing_labels = reshape_data(testing_images, testing_labels)

our_model = build_network(images)
x = 10
print(images[x].shape)
img = np.array(images[x]).reshape(IMG_SIZE, IMG_SIZE)
show(img)
exit()
our_model_trained = train_model(our_model, images, labels)

loss, acc = our_model_trained.evaluate(testing_images, testing_labels, batch_size=BATCHSIZE, use_multiprocessing='True')

acc = round(acc * 100, 2)

# prints the elapsed time for convenience
total_time = t.time() - start_time
total_time = round(total_time, 2)
total_time = convert_time(total_time)

model_results = f'''
NUMLAYERS = {NUMLAYERS}
NUMNODES = {NUMNODES}
NUMEPOCHS = {NUMEPOCHS}
BATCHSIZE = {BATCHSIZE}
IMG_SIZE = {IMG_SIZE}
ACCURACY = {acc}%
TIME = {total_time}
------------------------------------
'''


#print the results into results.txt
file = open('results.txt', 'a')
file.write(model_results)
file.close()


# final message
print(f"Finished in: {total_time}")
print('Success!')
