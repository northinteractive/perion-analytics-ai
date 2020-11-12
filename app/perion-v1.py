import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

# Get the dataset from googleapi

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

# image_count = len(list(data_dir.glob('*/*.jpg')))
# print(image_count)

batch_size = 32
img_height = 180
img_width = 180

# Create training Dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, 
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Create validation Dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# set class names from datasets attributes
class_names = train_ds.class_names

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

# configure the dataset for performance
# .cache keeps the images in memory after they're loaded off disk during first epoch
# .prefetch overlaps data preprocessing and model execution while training.

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Standarize - rescale [0,255] to [0,1] to standardize values for neural network
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

# Create the sequential model

num_classes = 5

model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

# Compile the model (using Adam Optimizer and losses.SparseCategoricalCrossentropy).
# To view training and validation for each metric, pass metrics.

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

# Get the model Summary

model.summary()

# Lets Train the model

epochs=10
history = model.fit(train_ds, validation_data=val_ds,epochs=epochs)

# print(class_names)

# import matplotlib.pyplot as plt 
# plt.figure(figsize=(10,10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3,3,i+1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")