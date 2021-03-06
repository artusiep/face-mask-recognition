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
"""Trains a Keras model to predict income bracket from other Census data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf

from . import model
from . import mark_done
from . import util


def get_args():
    """Argument parser.

    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting '
             'models')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=10,
        help='number of times to go through the data, default=20')
    parser.add_argument(
        '--batch-size',
        default=128,
        type=int,
        help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--learning-rate',
        default=.01,
        type=float,
        help='learning rate for gradient descent, default=.01')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    args, _ = parser.parse_known_args()
    return args


class CustomCallback(tf.keras.callbacks.TensorBoard):
    """Callback to write out a custom metric used by CAIP for HP Tuning."""

    def on_epoch_end(self, epoch, logs=None):  # pylint: disable=no-self-use
        """Write tf.summary.scalar on epoch end."""
        tf.summary.scalar('epoch_accuracy', logs['accuracy'], epoch)


def train_and_evaluate(args):
    """Trains and evaluates the Keras model.

    Uses the Keras model defined in model.py and trains on data loaded and
    preprocessed in util.py. Saves the trained model in TensorFlow SavedModel
    format to the path defined in part by the --job-dir argument.

    Args:
      args: dictionary of arguments - see get_args() for details
    """
    train_x, train_labels, eval_x, eval_labels = util.load_data('face_mask_recognition_dataset')

    dimensions = util.get_number_of_labels(train_labels)
    num_train_examples = len(train_x)
    # Create the Keras Model
    print(f"Number of dimensions: {dimensions}, Number of train examples {num_train_examples}")
    keras_model = model.create_basic_keras_model(dimensions)

    # # Pass a numpy array by passing DataFrame.values
    # training_dataset = model.input_fn(
    #     features=train_x.values,
    #     labels=train_y,
    #     shuffle=True,
    #     num_epochs=args.num_epochs,
    #     batch_size=args.batch_size)
    #
    # # Pass a numpy array by passing DataFrame.values
    # validation_dataset = model.input_fn(
    #     features=eval_x.values,
    #     labels=eval_y,
    #     shuffle=False,
    #     num_epochs=args.num_epochs,
    #     batch_size=num_eval_examples)

    # Setup Learning Rate decay.
    lr_decay_cb = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: args.learning_rate + 0.02 * (0.5 ** (1 + epoch)),
        verbose=True)

    # Setup TensorBoard callback.
    custom_cb = CustomCallback(os.path.join(args.job_dir, 'metric_tb'), histogram_freq=1)

    # Train model
    keras_model.fit(
        train_x, train_labels,
        steps_per_epoch=int(num_train_examples / args.batch_size),
        epochs=args.num_epochs,
        validation_data=(eval_x, eval_labels),
        validation_steps=1,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=[lr_decay_cb, custom_cb])

    export_path = os.path.join(args.job_dir, 'keras_export')
    tf.keras.models.save_model(keras_model, export_path)
    mark_done.mark_done(export_path)
    print('Model exported to: {}'.format(export_path))


if __name__ == '__main__':
    args = get_args()
    tf.compat.v1.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)