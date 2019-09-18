# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the ImageNet dataset using custom training loops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import time


from absl import app as absl_app
from absl import flags
from absl import logging
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from official.resnet.ctl import ctl_common
from official.vision.image_classification import cifar_preprocessing
from official.vision.image_classification import common
from official.vision.image_classification import resnet_cifar_model
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers


LR_SCHEDULE = [  # (multiplier, epoch to start) tuples
    (0.1, 91), (0.01, 136), (0.001, 182)
]

# Defines adjustable hyperparameters with defaults
HP_BATCH_SIZE = hp.HParam('batch_size')
HP_XLA = hp.HParam('xla')
HP_LR = hp.HParam('lr')
HP_FP16 = hp.HParam('fp16')
HP_TF_FUNCTION = hp.HParam('tf_function')
HP_SINGLE_L2_LOSS_OP = hp.HParam('single_l2_loss_op')

class TimeHistory(tf.keras.callbacks.Callback):
  """Callback for Keras models."""

  def __init__(self, batch_size, log_steps):
    """Callback for logging performance.

    Args:
      batch_size: Total batch size.
      log_steps: Interval of steps between logging of batch level stats.
    """
    self.batch_size = batch_size
    super(TimeHistory, self).__init__()
    self.log_steps = log_steps
    self.global_steps = 0
    self.examples_per_second = 0

  def on_train_begin(self, logs=None):
    self.train_start_time = time.time()

  def on_train_end(self, logs=None):
    self.train_total_time = time.time() - self.train_start_time

  def on_batch_begin(self, batch, logs=None):
    self.global_steps += 1
    if self.global_steps == 1:
      self.start_time = time.time()

  def on_batch_end(self, batch, logs=None):
    """Records elapse time of the batch and calculates examples per second."""
    if self.global_steps % self.log_steps == 0:
      timestamp = time.time()
      elapsed_time = timestamp - self.start_time
      self.examples_per_second = ((self.batch_size * self.log_steps) /
                                  elapsed_time)
      tf.compat.v1.logging.info(
          "BenchmarkMetric: {'global step':%d, 'time_taken': %f,"
          "'examples_per_second': %f}" %
          (self.global_steps, elapsed_time, self.examples_per_second))
      self.start_time = timestamp


def learning_rate_schedule(current_epoch, hparams):
  """Handles linear scaling rule and LR decay.

  Scale learning rate at epoch boundaries provided in LR_SCHEDULE by the
  provided scaling factor.

  Args:
    current_epoch: integer, current epoch indexed from 0.
    batch_size: integer, total batch sized.

  Returns:
    Adjusted learning rate.
  """
  if hparams[HP_LR] == 'scaled':
    initial_learning_rate = (common.BASE_LEARNING_RATE * hparams[HP_BATCH_SIZE]
                             / 128)
  else:
    initial_learning_rate = common.BASE_LEARNING_RATE
  learning_rate = initial_learning_rate
  for mult, start_epoch in LR_SCHEDULE:
    if current_epoch >= start_epoch:
      learning_rate = initial_learning_rate * mult
    else:
      break
  return learning_rate


def get_input_dataset(flags_obj, strategy, hparams):
  """Returns the test and train input datasets."""
  dtype = flags_core.get_tf_dtype(flags_obj)
  input_fn = cifar_preprocessing.input_fn

  train_ds = input_fn(
      is_training=True,
      data_dir=flags_obj.data_dir,
      batch_size=hparams[HP_BATCH_SIZE],
      parse_record_fn=cifar_preprocessing.parse_record,
      dtype=dtype)
  train_ds = strategy.experimental_distribute_dataset(train_ds)

  test_ds = input_fn(
      is_training=False,
      data_dir=flags_obj.data_dir,
      batch_size=hparams[HP_BATCH_SIZE],
      parse_record_fn=cifar_preprocessing.parse_record,
      dtype=dtype)
  test_ds = strategy.experimental_distribute_dataset(test_ds)

  return train_ds, test_ds


def hparam_run(flags_obj):

  HP_METRIC_ACCURACY = hp.Metric('top_1_accuracy', display_name='Accuracy')
  HP_METRIC_TOTAL_TIME = hp.Metric('total_time', display_name='Total Time')
  HP_METRIC_EXP_PER_SEC = hp.Metric('exp_per_sec', display_name='Exp/Sec')

  global HP_BATCH_SIZE, HP_XLA, HP_LR, HP_FP16, HP_TF_FUNCTION, HP_SINGLE_L2_LOSS_OP
  HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([128]))
  HP_XLA = hp.HParam('xla', hp.Discrete([False]))
  HP_LR = hp.HParam('lr', hp.Discrete(['scaled']))
  HP_FP16 = hp.HParam('fp16', hp.Discrete([False]))
  HP_TF_FUNCTION = hp.HParam('tf_function', hp.Discrete([True]))

  with tf.summary.create_file_writer(flags_obj.model_dir).as_default():
    hp.hparams_config(
        hparams=[HP_BATCH_SIZE, HP_XLA, HP_LR, HP_FP16, HP_TF_FUNCTION,
                 HP_SINGLE_L2_LOSS_OP],
        metrics=[HP_METRIC_ACCURACY, HP_METRIC_TOTAL_TIME,
                 HP_METRIC_EXP_PER_SEC],
    )

  hparam_lists = [HP_BATCH_SIZE.domain.values,
                  HP_XLA.domain.values,
                  HP_LR.domain.values,
                  HP_FP16.domain.values,
                  HP_TF_FUNCTION.domain.values]

  hparam_product = list(itertools.product(*hparam_lists))
  i = 0
  for combination in hparam_product:
    print('Running {} of {} combinations'.format(i, len(hparam_product)))
    print(combination)
    hparams = {
        HP_BATCH_SIZE: combination[0],
        HP_XLA: combination[1],
        HP_LR: combination[2],
        HP_FP16: combination[3],
        HP_TF_FUNCTION: combination[4],
    }
    run(flags_obj, hparams, hparam_dir='hparam_combo_{}'.format(i))
    i += 1


def run(flags_obj, hparams, hparam_dir=None):
  """Run ResNet ImageNet training and eval loop using custom training loops.

  Args:
    flags_obj: An object containing parsed flag values.

  Raises:
    ValueError: If fp16 is passed as it is not currently supported.

  Returns:
    Dictionary of training and eval stats.
  """
  if hparams[HP_XLA]:
    tf.config.optimizer.set_jit(True)
    # Disable PinToHostOptimizer in grappler when enabling XLA because it
    # causes OOM and performance regression.
    tf.config.optimizer.set_experimental_options(
        {'pin_to_host_optimization': False}
    )

  # Sets summary output directories for tensorboard.
  train_log_dir = os.path.join(flags_obj.model_dir, hparam_dir)
  train_summary_writer = tf.summary.create_file_writer(train_log_dir)
  with train_summary_writer.as_default():
    hp.hparams(hparams)  # record the values used in this trial

  tf.keras.backend.set_image_data_format('channels_first')

  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy='default',
      num_gpus=flags_obj.num_gpus)

  train_ds, test_ds = get_input_dataset(flags_obj, strategy, hparams)

  train_steps = (
      cifar_preprocessing.NUM_IMAGES['train'] // hparams[HP_BATCH_SIZE])

  eval_steps = (
      cifar_preprocessing.NUM_IMAGES['validation'] // hparams[HP_BATCH_SIZE])

  train_epochs = flags_obj.train_epochs

  time_callback = TimeHistory(hparams[HP_BATCH_SIZE], 50)

  strategy_scope = distribution_utils.get_strategy_scope(strategy)
  with strategy_scope:
    model = resnet_cifar_model.resnet56(
        classes=cifar_preprocessing.NUM_CLASSES)

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=common.BASE_LEARNING_RATE, momentum=0.9,
        nesterov=True)

    if hparams[HP_FP16]:
      if not hparams[HP_TF_FUNCTION]:
        raise ValueError('FP16 requires tf_function to be true')
      optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
          optimizer, 'dynamic')

    training_accuracy = tf.keras.metrics.CategoricalAccuracy(
        'training_accuracy', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(
        'test_accuracy', dtype=tf.float32)

    trainable_variables = model.trainable_variables

    def train_step(train_ds_inputs):
      """Training StepFn."""
      def step_fn(inputs):
        """Per-Replica StepFn."""
        images, labels = inputs
        with tf.GradientTape() as tape:
          logits = model(images, training=True)

          prediction_loss = tf.keras.losses.categorical_crossentropy(
              labels, logits)
          loss = tf.reduce_sum(prediction_loss) * (1.0/ hparams[HP_BATCH_SIZE])
          num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
          loss += (tf.reduce_sum(model.losses) / num_replicas)

          # Scale the loss
          if hparams[HP_FP16]:
            loss = optimizer.get_scaled_loss(loss)

        grads = tape.gradient(loss, trainable_variables)

        # Unscale the grads
        if hparams[HP_FP16]:
          grads = optimizer.get_unscaled_gradients(grads)

        optimizer.apply_gradients(zip(grads, trainable_variables))

        training_accuracy.update_state(labels, logits)
        return loss

      per_replica_losses = strategy.experimental_run_v2(
          step_fn, args=(train_ds_inputs,))
      return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                             axis=None)

    def test_step(test_ds_inputs):
      """Evaluation StepFn."""
      def step_fn(inputs):
        images, labels = inputs
        logits = model(images, training=False)
        loss = tf.keras.losses.categorical_crossentropy(labels, logits)
        loss = tf.reduce_sum(loss) * (1.0/ hparams[HP_BATCH_SIZE])
        test_loss.update_state(loss)
        test_accuracy.update_state(labels, logits)

      if strategy:
        strategy.experimental_run_v2(step_fn, args=(test_ds_inputs,))
      else:
        step_fn(test_ds_inputs)

    if hparams[HP_TF_FUNCTION]:
      train_step = tf.function(train_step)
      test_step = tf.function(test_step)

    time_callback.on_train_begin()
    for epoch in range(train_epochs):

      train_iter = iter(train_ds)
      total_loss = 0.0
      training_accuracy.reset_states()

      for step in range(train_steps):
        optimizer.lr = learning_rate_schedule(epoch, hparams)

        time_callback.on_batch_begin(step+epoch*train_steps)
        total_loss += train_step(next(train_iter))
        time_callback.on_batch_end(step+epoch*train_steps)

      train_loss = total_loss / train_steps
      with train_summary_writer.as_default():
        tf.summary.scalar('train_accuracy', training_accuracy.result(),
                          step=epoch)
        tf.summary.scalar('train_loss', train_loss, step=epoch)
        #tf.summary.scalar('lr', optimizer.lr, step=epoch)
        #tf.summary.scalar('exp_per_sec', time_callback.examples_per_second,
        #                  step=epoch)

      logging.info('Training loss: %s, accuracy: %s%% at epoch: %d',
                   train_loss.numpy(),
                   training_accuracy.result().numpy(),
                   epoch)

      test_loss.reset_states()
      test_accuracy.reset_states()

      test_iter = iter(test_ds)
      for _ in range(eval_steps):
        test_step(next(test_iter))
      with train_summary_writer.as_default():
        tf.summary.scalar('top_1_accuracy', test_accuracy.result(),
                          step=epoch)
      logging.info('Test loss: %s, accuracy: %s%% at epoch: %d',
                   test_loss.result().numpy(),
                   test_accuracy.result().numpy(),
                   epoch)

    time_callback.on_train_end()
    with train_summary_writer.as_default():
      tf.summary.scalar('total_time', time_callback.train_total_time / 60,
                        step=0)


def main(_):
  model_helpers.apply_clean(flags.FLAGS)
  with logger.benchmark_context(flags.FLAGS):
    return hparam_run(flags.FLAGS)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  common.define_keras_flags()
  ctl_common.define_ctl_flags()
  flags.adopt_module_key_flags(ctl_common)
  absl_app.run(main)
