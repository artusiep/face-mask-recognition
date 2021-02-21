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
"""Utilities to download and preprocess the Census data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets
from six.moves import urllib


def _download_and_clean_file(filename, url):
    """Downloads data from url, and makes changes to match the CSV format.

    The CSVs may use spaces after the comma delimters (non-standard) or include
    rows which do not represent well-formed examples. This function strips out
    some of these problems.

    Args:
      filename: filename to save url to
      url: URL of resource to download
    """
    temp_file, _ = urllib.request.urlretrieve(url)
    with tf.io.gfile.GFile(temp_file, 'r') as temp_file_object:
        with tf.io.gfile.GFile(filename, 'w') as file_object:
            for line in temp_file_object:
                line = line.strip()
                line = line.replace(', ', ',')
                if not line or ',' not in line:
                    continue
                if line[-1] == '.':
                    line = line[:-1]
                line += '\n'
                file_object.write(line)
    tf.io.gfile.remove(temp_file)


def image_rgb_to_gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def preprocess(dataframe):
    """

    Args:
      dataframe: Pandas dataframe with raw data

    Returns:
      Dataframe with preprocessed data
    """
    # Convert integer valued (numeric) columns to floating point
    gray = np.array([image_rgb_to_gray(rgb) for rgb in dataframe])
    shape = gray.shape
    gray = gray.reshape(*shape, 1)
    return gray


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Scales numerical columns using their means and standard deviation to get
    z-scores: the mean of each numerical column becomes 0, and the standard
    deviation becomes 1. This can help the model converge during training.

    Args:
      images: np images

    Returns:
      Input dataframe with the numerical columns scaled to z-scores
    """
    return images.astype('float32')


def normalize_labels(labels: np.ndarray) -> np.ndarray:
    return labels.astype('float32')


def get_df_images_and_labels(tensorflow_ds: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    images = np.array([example['image'].numpy() for example in tensorflow_ds])
    labels = np.array([example['label'].numpy() for example in tensorflow_ds])
    return images, labels


def get_number_of_labels(train_labels: np.ndarray) -> int:
    return len(np.unique(train_labels))


def datasets_info(variable_name, df):
    print(f"{variable_name} dataset. Shape: {df.shape}")


def load_data(tensorflow_dataframe_name) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads data into preprocessed (train_x, train_labels, eval_y, eval_labels)
    dataframes.

    Returns:
      A tuple (train_x, train_y, eval_x, eval_y), where train_x and eval_x are
      Pandas dataframes with features for training and train_y and eval_y are
      numpy arrays with the corresponding labels.
    """
    # noinspection PyUnresolvedReferences
    from . import face_mask_recognition_dataset
    dl_config = tensorflow_datasets.download.DownloadConfig(register_checksums=True)

    train, eval = tensorflow_datasets.load(tensorflow_dataframe_name, split=['train', 'test'],
                                           download_and_prepare_kwargs={'download_config': dl_config})
    train_df, train_labels = get_df_images_and_labels(train)
    eval_df, eval_labels = get_df_images_and_labels(eval)

    train_df = preprocess(train_df)
    eval_df = preprocess(eval_df)

    train_x = normalize_images(train_df)
    eval_x = normalize_images(eval_df)
    train_labels = normalize_labels(train_labels)
    eval_labels = normalize_labels(eval_labels)

    datasets_info("train_x", train_x)
    datasets_info("train_labels", train_labels)
    datasets_info("eval_x", eval_x)
    datasets_info("eval_labels", eval_labels)

    return train_x, train_labels, eval_x, eval_labels
