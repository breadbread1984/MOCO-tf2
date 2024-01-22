#!/usr/bin/python3

import os;
import numpy as np;
import tensorflow as tf;
import tensorflow_datasets as tfds;
from models import Encoder, RandomAugmentation, Queue;
from download_dataset import parse_function;

batch_size = 100;
input_shape = (64, 64, 3);
temp = 1; #0.07;
beta = 0.999;

def main():

  # query and key feature extractor
  f_q = Encoder(input_shape); # update this model more frequently
  f_k = Encoder(input_shape); # update this model less frequently
  for source, target in zip(f_q.trainable_weights, f_k.trainable_weights):
    target.assign(source)
  # utils for training
  optimizer = tf.keras.optimizers.SGD(0.001, momentum = 0.9, decay = 0.0001);
  trainset = iter(tfds.load(name = 'imagenet_resized/64x64', split = tfds.Split.TRAIN, download = False).repeat(-1).map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE));
  checkpoint = tf.train.Checkpoint(f_q = f_q, f_k = f_k, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  log = tf.summary.create_file_writer('checkpoints');
  avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
  # stuff 10 batches feature into queue
  queue = Queue(trainset, f_k, 10);
  augmentation = RandomAugmentation(input_shape, rotation_range = (-10, 10));
  while True:
    x, label = next(trainset);
    # two augmented versions of the same batch data
    x_q = augmentation(x); # x_q.shape = (batch, 64, 64, 3)
    x_k = augmentation(x); # x_k.shape = (batch, 64, 64, 3)
    with tf.GradientTape() as tape:
      q = f_q(x_q); # q.shape = (batch, 128)
      k = f_k(x_k); # k.shape = (batch, 128)
      l_pos = tf.reshape(tf.linalg.matmul(tf.reshape(q, (-1, 1, 128)), tf.reshape(k, (-1, 128, 1))), (-1, 1)); # l_pos.shape = (batch, 1)
      l_neg = tf.reshape(tf.linalg.matmul(tf.reshape(q, (-1, 1, 128)), queue.get()), (-1, 10)); # l_neg.shape = (batch, 10)
      logits = tf.concat([l_pos, l_neg], axis = 1); # logits.shape = (batch, 11)
      # contrastive loss
      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)(tf.zeros((batch_size,)), logits / temp);
    grads = tape.gradient(loss, f_q.trainable_variables); avg_loss.update_state(loss);
    [tf.debugging.Assert(tf.math.logical_not(tf.math.reduce_any(tf.math.is_nan(grad))), grads + [optimizer.iterations,]) for grad in grads];
    [tf.debugging.Assert(tf.math.logical_not(tf.math.reduce_any(tf.math.is_inf(grad))), grads + [optimizer.iterations,]) for grad in grads];
    tf.debugging.Assert(tf.math.logical_not(tf.math.reduce_any(tf.math.is_nan(f_q(tf.constant(np.random.normal(size = (1, 64, 64, 3)), dtype = tf.float32))))), [optimizer.iterations]);
    optimizer.apply_gradients(zip(grads, f_q.trainable_variables));
    # momentum update
    tf.debugging.Assert(tf.math.logical_not(tf.math.reduce_any(tf.math.is_nan(f_q(tf.constant(np.random.normal(size = (1, 64, 64, 3)), dtype = tf.float32))))), [optimizer.iterations]);
    for i in range(len(f_q.trainable_variables)):
      f_k.trainable_variables[i].assign(beta * f_k.trainable_variables[i] + (1 - beta) * f_q.trainable_variables[i]);
    # update dictionary
    queue.update(k);
    # write log
    if tf.equal(optimizer.iterations % 500, 0):
      with log.as_default():
        tf.summary.scalar('loss', avg_loss.result(), step = optimizer.iterations);
      print('Step #%d Loss: %.6f' % (optimizer.iterations, avg_loss.result()));
      avg_loss.reset_states();
    if tf.equal(optimizer.iterations % 5000, 0):
      # save model
      checkpoint.save(os.path.join('checkpoints', 'ckpt'));
      if False == os.path.exists('models'): os.mkdir('models');
      f_k.save(os.path.join('models', 'model.h5'));

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();
