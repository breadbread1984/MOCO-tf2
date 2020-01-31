#!/usr/bin/python3

from math import radians;
import tensorflow as tf;
from AffineLayer import AffineLayer;

def Encoder(input_shape):

  inputs = tf.keras.Input(input_shape[-3:]);
  model = tf.keras.applications.ResNet50(input_tensor = inputs, include_top = False);
  results = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis = (1, 2)))(model.outputs[0]); # results.shape = (batch, 2048)
  results = tf.keras.layers.Dense(units = 128)(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def RandomAffine(input_shape, rotation_range = (0, 0), scale_range = (1, 1), translate_range = (0, 0)):

  inputs = tf.keras.Input(input_shape[-3:]);
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

def RandomAugmentation(input_shape, rotation_range = (-20, 20), scale_range = (0.8, 1.2), padding = 10, bri = 32./255., sat = (0.5, 1.5), hue = .2, con = (0.5, 1.5)):

  inputs = tf.keras.Input(input_shape[-3:]);
  results = RandomAffine(inputs.shape[-3:], rotation_range = rotation_range, scale_range = scale_range)(inputs);
  results = tf.keras.layers.Lambda(lambda x, p: tf.image.resize(x, [tf.shape(x)[-3] + p, tf.shape(x)[-2] + p], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR), arguments = {'p': padding})(results);
  results = tf.keras.layers.Lambda(lambda x: tf.image.random_crop(x[0], size = tf.shape(x[1])))([results, inputs]);
  results = tf.keras.layers.Lambda(lambda x: tf.image.random_flip_left_right(x))(results);
  results = tf.keras.layers.Lambda(lambda x, b: tf.image.random_brightness(x, b), arguments = {'b': bri})(results);
  results = tf.keras.layers.Lambda(lambda x, a, b: tf.image.random_saturation(x, lower = a, upper = b), arguments = {'a': sat[0], 'b': sat[1]})(results);
  results = tf.keras.layers.Lambda(lambda x, h: tf.image.random_hue(x, h), arguments = {'h': hue})(results);
  results = tf.keras.layers.Lambda(lambda x, a, b: tf.image.random_contrast(x, lower = a, upper = b), arguments = {'a': con[0], 'b': con[1]})(results);
  results = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, 0., 1.))(results);
  results = tf.keras.layers.Lambda(lambda x: tf.math.multiply(tf.math.subtract(x, 0.5), 2.))(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

class Queue(object):

  def __init__(self, trainset, encoder, batches = 10):

    self.queue = list();
    for i in range(batches):
      data, _ = next(trainset);
      keys = encoder(data); # keys.shape = (batch, 128)
      self.queue.append(keys);

  def update(self, feature):
      
    del self.queue[0];
    self.queue.append(feature);

  def get(self):

    data = tf.stack(self.queue, axis = -1) # data.shape = (batch, 128, 10)
    return data;

if __name__ == "__main__":

  assert True == tf.executing_eagerly();
  import cv2;
  import tensorflow_datasets as tfds;
  from download_dataset import parse_function;
  trainset = iter(tfds.load(name = 'imagenet_resized/64x64', split = tfds.Split.TRAIN, download = False).map(parse_function).batch(8));
  images, _ = next(trainset);
  for j in range(2):
    augments = RandomAugmentation(images.shape[-3:], rotation_range = (-10, 10))(images);
    for i in range(8):
      image = tf.cast(tf.clip_by_value(images[i, ...] * 255., 0., 255.), dtype = tf.uint8);
      augment = tf.cast(tf.clip_by_value((augments[i, ...] + 1) * 127.5, 0., 255.), dtype = tf.uint8);
      image = tf.concat([image,augment], axis = 1).numpy();
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR);
      cv2.imshow(str(i), image);
    cv2.waitKey();

