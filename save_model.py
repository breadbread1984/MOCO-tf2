#!/usr/bin/python3

import os;
import tensorflow as tf;
from models import Encoder;

input_shape = (64, 64, 3);

def save_model():

  f_k = Encoder(input_shape);
  optimizer = tf.keras.optimizers.Adam(0.01, decay = 0.0001);
  checkpoint = tf.train.Checkpoint(model = f_k, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  if False == os.path.exists('models'): os.mkdir('models');
  f_k.save(os.path.join('models', 'resnet50.h5'));

if __name__ == "__main__":

  assert tf.executing_eagerly();
  save_model();
