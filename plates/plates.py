"""
Author: Salman Khan Pathan

Credits: This is from the course https://www.udemy.com/course/deeplearning/
"""
# Loading modules
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# Data pre-processing
# These transformations are primarily to avoid overfitting
# as we include more cases by using shear, zoom and flip
train_datagen = ImageDataGenerator(rescale = 1./255,  # Feature scaling
                                   # shear_range = 0.2,
                                   zoom_range = 0.2,
                                   height_shift_range=0.2,
                                   width_shift_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True)
# Image augmentation
# This takes the files from our filesystem and applies the target size
# This also divides the data into batches to be used later
# This batch size is also the size of batches in our model evaluation
# Batch size determination is more of a trial and error process
training_set = train_datagen.flow_from_directory(
    'C:\\Users\\patha\\Downloads\\platesv2\\plates\\plates\\train',
    target_size = (64, 64),
    # batch_size = 4,
    class_mode = 'binary')
print(training_set.class_indices)
# Creates data in batches
# Only rescaling in test set (and not other tranformations), to avoid any
# leakage
# test_datagen = ImageDataGenerator(rescale = 1./255)
# # This takes the files from our filesystem and applies the target size
# test_set = test_datagen.flow_from_directory(
#     'C:\\Users\\patha\\Downloads\\platesv2\\plates\\plates\\test',
#     target_size = (64, 64),
#     batch_size = 32,
#     class_mode = 'binary')

# Building CNN
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=11, activation='relu',
                               input_shape=(64, 64, 3)))
# In this structure, tf automatically connects the subsequent layers with
# the previous ones
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=1))
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=7, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=1))
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=1))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Training CNN
# Can use CategoricalCrossentropy for multi-class problems
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy',
            metrics = ['accuracy'])
cnn.fit(x=training_set, epochs=100, verbose=2)
