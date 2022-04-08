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
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
# Image augmentation
# This takes the files from our filesystem and applies the target size
# This also divides the data into batches to be used later
# This batch size is also the size of batches in our model evaluation
# Batch size determination is more of a trial and error process
training_set = train_datagen.flow_from_directory(
    '<path-to-training-data-here>',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary')
print(training_set.class_indices)
# Creates data in batches
# Only rescaling in test set (and not other tranformations), to avoid any
# leakage
test_datagen = ImageDataGenerator(rescale = 1./255)
# This takes the files from our filesystem and applies the target size
test_set = test_datagen.flow_from_directory(
    '<path-to-test-data-here>',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary')

# Building CNN
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',
                               input_shape=(64, 64, 3)))
# In this structure, tf automatically connects the subsequent layers with
# the previous ones
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Training CNN
# Can use CategoricalCrossentropy for multi-class problems
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy',
            metrics = ['accuracy'])
cnn.fit(x=training_set, validation_data=test_set, epochs=25)
# Training accuracy: 90%. Validation accuracy: 81%
# Runtime with GPU: 23s per epoch. Without GPU: 16s per epoch
# TODO: Check validation accuracy with and without augmentation

# Prediction
test_image = image.load_img('<single-image-path-here>',
                            target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
if result[0][0] > 0.5:
    print('Dog')
else:
    print('Cat')
