#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_datasets as tfds;

def parse_function(feature):

  data = tf.cast(feature["image"], dtype = tf.float32) / 255.;
  label = feature["label"];
  return data, label;

def download():

  # load dataset
  dataset_builder = tfds.builder('imagenet_resized/64x64');
  dataset_builder.download_and_prepare();
  # try to load the dataset once
  trainset = tfds.load(name = 'imagenet_resized/64x64', split = tfds.Split.TRAIN, download = False);
  testset = tfds.load(name = 'imagenet_resized/64x64', split = tfds.Split.VALIDATION, download = False);

if __name__ == "__main__":

  assert tf.executing_eagerly();
  download();

