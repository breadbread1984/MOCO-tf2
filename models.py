#!/usr/bin/python3

from math import radians;
import tensorflow as tf;
from AffineLayer import AffineLayer;

def RandomAffine(input_shape, rotation_range = (0, 0), scale_range = (1, 1), translate_range = (0, 0)):

  inputs = tf.keras.Input(input_shape);
  theta = tf.keras.layers.Lambda(lambda x, a, b: tf.random.uniform((tf.shape(x)[0], 1, 1), minval = radians(a), maxval = radians(b), dtype = tf.float32), arguments = {'a': rotation_range[0], 'b': rotation_range[1]})(inputs);
  scale = tf.keras.layers.Lambda(lambda x, a, b: tf.random.uniform((tf.shape(x)[0], 1, 1), minval = a, maxval = b, dtype = tf.float32), arguments = {'a': scale_range[0], 'b': scale_range[1]})(inputs);
  t1 = tf.keras.layers.Lambda(lambda x, t: tf.random.uniform((tf.shape(x)[0], 1, 1), minval = 0, maxval = t, dtype = tf.float32), arguments = {'t': translate_range[0]})(inputs);
  t2 = tf.keras.layers.Lambda(lambda x, t: tf.random.uniform((tf.shape(x)[0], 1, 1), minval = 0, maxval = t, dtype = tf.float32), arguments = {'t': translate_range[1]})(inputs);
  beta = tf.keras.layers.Lambda(lambda x: x[0] * tf.math.sin(x[1]))([scale, theta]); # sin.shape = (batch, 1, 1)
  alpha = tf.keras.layers.Lambda(lambda x: x[0] * tf.math.cos(x[1]))([scale, theta]); # cos.shape = (batch, 1, 1)
  row1 = tf.keras.layers.Concatenate(axis = -1)([alpha, beta, t1]); # row1.shape = (batch, 1, 3)
  row2 = tf.keras.layers.Concatenate(axis = -1)([-beta, alpha, t2]); # row2.shape = (batch, 1, 3)
  affines = tf.keras.layers.Concatenate(axis = -2)([row1, row2]); # affine.shape = (batch, 2, 3);
  outputs = AffineLayer()(inputs,affines);
  return tf.keras.Model(inputs = inputs, outputs = outputs);



if __name__ == "__main__":

  assert True == tf.executing_eagerly();
  import cv2;
  (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data();
  img = train_x[0,...];
  inputs = tf.expand_dims(tf.constant(img), axis = 0); # add batch
  inputs = tf.expand_dims(inputs, axis = -1); # add channel
  affined = RandomAffine(inputs.shape[-3:], rotation_range = (20,20), scale_range = (1,1), translate_range = (1.1,1.1))(inputs);
  cv2.imshow('affined',tf.cast(affined[0,...,0], dtype = tf.uint8).numpy());
  cv2.waitKey();
