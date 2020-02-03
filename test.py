#!/usr/bin/python3

import os;
import tensorflow as tf;
import tensorflow_datasets as tfds;
from download_dataset import parse_function;
from SKL import SKL;

batch_size = 100;

def main():

  testset = tfds.load(name = 'imagenet_resized/64x64', split = tfds.Split.VALIDATION, download = False).map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  encoder = tf.keras.models.load_model(os.path.join('models', 'resnet50.h5'), compile = False, custom_objects = {'tf': tf});
  skl = SKL(128, 2);
  for data, _ in testset:
    code = encoder(data);
    eigvec, sigma, mean = skl(code); # eigvec.shape = (128, 2)
  testset = tfds.load(name = 'imagenet_resized/64x64', split = tfds.Split.VALIDATION, download = False).map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  low_dim = tf.constant((0, 2), dtype = tf.float32);
  labels = tf.constant((0, 1), dtype = tf.int64);
  for data, label in testset:
    code = encoder(data); # code.shape = (100, 128)
    code = tf.linalg.matmul(code, eigvec); # code.shape = (100,2)
    low_dim = tf.concat([low_dim, code], axis = 0);
    labels = tf.concat([labels, label], axis = 0);
  # TODO: plot data points

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();
