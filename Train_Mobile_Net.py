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

