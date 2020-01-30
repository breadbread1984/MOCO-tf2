#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_datasets as tfds;

def parse_function(feature):

  data = feature["image"];
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
  import cv2;
  trainset = iter(tfds.load(name = 'imagenet_resized/64x64', split = tfds.Split.TRAIN, download = False));
  image, _ = next(trainset);
  for i in range(5):
    augmented = RandomAugmentation(image.shape[-3:], rotation_range = (-10, 10))(image);
    augmented = tf.cast(tf.clip_by_value((augmented + 1) * 127.5, 0., 255.)[0], dtype = tf.uint8);
    cv2.imshow(str(i), augmented);
  cv2.waitKey();
