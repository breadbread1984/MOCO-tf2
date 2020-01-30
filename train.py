#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_datasets as tfds;
from models import Encoder, RandomAugmentation, Queue;
from download_dataset import parse_function;

batch_size = 100;
input_shape = (64, 64, 3);

def main():

  model = Encoder(input_shape);
  optimizer = tf.keras.optimizers.Adam(0.01, decay = 0.0001);
  trainset = iter(tfds.load(name = 'imagenet_resized/64x64', split = tfds.Split.TRAIN, download = False).repeat(-1).map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE));
  checkpoint = tf.train.Checkpoint(model = model, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  log = tf.summary.create_file_writer('checkpoints');
  avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
  # feature queue
  q = Queue(trainset, model);
  augmentation = RandomAugmentation(input_shape, rotation_range = (-10, 10));
  while True:
    image, label = next(trainset);
    augmented = augmentation(image);
    import cv2;
    for i in range(5):
      outputs = tf.cast(tf.clip_by_value((augmented[i,...] + 1) * 127.5, 0., 255.), dtype = tf.uint8);
      cv2.imshow(str(i), outputs.numpy());
    cv2.waitKey();

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();
