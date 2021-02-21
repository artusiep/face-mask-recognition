# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Defines a Keras model and input function for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras


# def input_fn(features, labels, shuffle, num_epochs, batch_size):
#     """Generates an input function to be used for model training.
#
#     Args:
#       features: numpy array of features used for training or inference
#       labels: numpy array of labels for each example
#       shuffle: boolean for whether to shuffle the data or not (set True for
#         training, False for evaluation)
#       num_epochs: number of epochs to provide the data for
#       batch_size: batch size for training
#
#     Returns:
#       A tf.data.Dataset that can provide data to the Keras model for training or
#         evaluation
#     """
#     if labels is None:
#         inputs = features
#     else:
#         inputs = (features, labels)
#     dataset = tf.data.Dataset.from_tensor_slices(inputs)
#
#     if shuffle:
#         dataset = dataset.shuffle(buffer_size=len(features))
#
#     # We call repeat after shuffling, rather than before, to prevent separate
#     # epochs from blending together.
#     dataset = dataset.repeat(num_epochs)
#     dataset = dataset.batch(batch_size)
#     return dataset


def create_basic_keras_model(dimensions: int):
    """Creates Basic Keras Model for image Classificaton.

    The single output node + Sigmoid activation makes this a Logistic
    Regression.

    Args:
      dimensions: How many features the input has

    Returns:
      The compiled Keras model (still needs to be trained)
    """
    model = keras.Sequential(
        [
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(100, 3, activation='relu'),
            keras.layers.MaxPool2D(2, 2),

            keras.layers.Conv2D(100, 3, activation='relu'),
            keras.layers.MaxPool2D(2, 2),

            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dense(dimensions, activation='softmax')
        ])

    optimizer = 'adam'
    print("Starting learning model")
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])
    return model
