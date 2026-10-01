# Exercise 4.9 VGG19.py
from keras.preprocessing import image
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
import numpy as np

model = VGG19(weights='imagenet')
print(model.summary())
