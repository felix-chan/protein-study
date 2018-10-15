import keras
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.utils import np_utils
from keras.layers import Dense, Dropout,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
import pandas as pd

model = Sequential()

K.set_image_data_format('channels_last')
K.tensorflow_backend._get_available_gpus()
np.random.seed(0)

# Load image and labels
import skimage.io
from skimage.transform import resize
import math
from keras.preprocessing import image

class load_data:
    
    # Create dataset
    def create_dataset(dataset_info, shape):
        batch_size = 10
        assert shape[2] == 4
        while True:
            random_indexes = np.random.choice(len(dataset_info), batch_size)
            batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))
            batch_labels = np.zeros((batch_size, 28))
            for i, idx in enumerate(random_indexes):
                image_array = load_data.load_image(
                    dataset_info[idx]['path'], shape)   
                batch_images[i] = image_array
                batch_labels[i][dataset_info[idx]['labels']] = 1
            yield batch_images, batch_labels
        
    def load_image(path, shape):
        image_red_ch = image.img_to_array(image.load_img(path+'_red.png', target_size=(shape[0], shape[1])))
        image_yellow_ch = image.img_to_array(image.load_img(path+'_yellow.png', target_size=(shape[0], shape[1])))
        image_green_ch = image.img_to_array(image.load_img(path+'_green.png', target_size=(shape[0], shape[1])))
        image_blue_ch = image.img_to_array(image.load_img(path+'_blue.png', target_size=(shape[0], shape[1])))

        image_set = np.stack((
            image_red_ch[:,:,0], 
            image_yellow_ch[:,:,0], 
            image_green_ch[:,:,0], 
            image_blue_ch[:,:,0]), -1)
        return image_set
        

import os

path_to_train = '../train/'
data = pd.read_csv('../train.csv')

train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    train_dataset_info.append({
        'path':os.path.join(path_to_train, name),
        'labels':np.array([int(label) for label in labels])})
train_dataset_info = np.array(train_dataset_info)

# Create train and validating dataset
indexes = np.arange(train_dataset_info.shape[0])

np.random.shuffle(indexes)
train_size = math.floor(len(train_dataset_info) * 0.66)
# train_indexes = indexes[:train_size]
# valid_indexes = indexes[train_size:]

train_indexes = indexes[:100]
valid_indexes = indexes[100:200]

# create train and valid datagens

train_generator = load_data.create_dataset(
    train_dataset_info[train_indexes], (256,256,4))
validation_generator = load_data.create_dataset(
    train_dataset_info[valid_indexes], (256,256,4))

# Design the model

from keras.layers import Dropout

# First layer
print("Start modelling")

model.add(Conv2D(10, kernel_size=5, padding="same",input_shape=(256,256,4), activation = 'relu'))
model.add(Conv2D(10, kernel_size=5, padding="valid", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

# Second layer

model.add(Conv2D(70, kernel_size=3, padding="same", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Third layer
model.add(Conv2D(100, kernel_size=3, padding="valid", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))

# Flatten all layers

from keras.layers.core import Activation

model.add(Flatten())
model.add(Dense(50))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(28))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# Fit model
from keras.callbacks import ModelCheckpoint

epochs = 50; batch_size = 16
checkpointer = ModelCheckpoint(
    '../working/InceptionResNetV2.model', 
    verbose=2, 
    save_best_only=True)

print("Start fitting model")
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    validation_data=next(validation_generator),
    epochs=epochs, 
    verbose=1,
    callbacks=[checkpointer])

