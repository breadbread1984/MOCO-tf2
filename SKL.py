#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;

@tf.function
def gram_schmidt(inputs):
  # inputs.shape = (number of vector, dimension of vector)
  time = tf.constant(1, dtype = tf.int32);
  def diagonalize(t, basis):
    alpha = basis[t:,...]; # alpha.shape = (N - t, D)
    beta = basis[t-1:t,...]; # beta.shape = (1, D)
    ab = tf.linalg.matmul(alpha, beta, transpose_b = True); # ab.shape = (N-t, 1)
    bb = tf.linalg.matmul(beta, beta, transpose_b = True); # bb.shape = (1,1)
    weight = tf.math.divide_no_nan(ab,bb); # weight.shape = (N-t, 1)
    weighted_b = tf.linalg.matmul(weight, beta); # weighted_b.shape = (N-t, D)
    subtracted = tf.math.subtract(alpha, weighted_b); # subtracted.shape = (N-t, D)
    basis = tf.concat([basis[:t,...], subtracted], axis = 0);
    t = tf.math.add(t, tf.ones_like(t, dtype = tf.int32));
    return t, basis;
  time, basis = tf.while_loop(lambda t, b: tf.math.less(t, tf.shape(b)[0]), diagonalize, loop_vars = [time, inputs], shape_invariants = [time.get_shape(), tf.TensorShape([None, inputs.shape[1]])]);
  norm = tf.norm(basis, ord = 2, axis = 1); # norm.shape = (N);
  basis = tf.math.divide(basis, tf.expand_dims(norm, axis = 1)); # basis.shape = (N, D)
  return basis;

def PCA(dimension, principal_num = 5):
  # inputs.shape = (number of vectors, dimension of vector)
  inputs = tf.keras.Input((dimension,));
  mean = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis = 0, keepdims = True))(inputs);
  centered = tf.keras.layers.Lambda(lambda x: x[0] - x[1])([inputs, mean]);
  centered = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1, 0)))(centered);
  s,u,v = tf.keras.layers.Lambda(lambda x: tf.linalg.svd(x))(centered);
  sigma = tf.keras.layers.Lambda(lambda x, n: x[...,:n], arguments = {'n': principal_num})(s);
  eigvec = tf.keras.layers.Lambda(lambda x, n: x[...,:n], arguments = {'n': principal_num})(u);
  return tf.keras.Model(inputs = inputs, outputs = (eigvec, sigma, mean));

class SKL(tf.keras.Model):

  def __init__(self, dimension, principal_num = 5):
      
    super(SKL, self).__init__();
    self.eigvec = None; # column vectors
    self.sigma = None;
    self.mean = None;
    self.dimension = dimension;
    self.principal_num = principal_num;
    self.pca = PCA(dimension, principal_num);

  def call(self, inputs):

    # inputs are row vectors
    assert len(inputs.shape) == 2 and inputs.shape[1] == self.dimension;
    if self.eigvec is None:
      self.eigvec, self.sigma, self.mean = self.pca(inputs);
    else:
      mean = tf.math.reduce_mean(inputs, axis = 0, keepdims = True)
      self.mean = 0.9 * self.mean + 0.1 * mean;
      centered = inputs - self.mean;
      residual = tf.transpose(centered, (1, 0)) - tf.linalg.matmul(tf.linalg.matmul(self.eigvec, self.eigvec, transpose_b = True), centered, transpose_b = True);
      orthogonalized = gram_schmidt(tf.transpose(residual, (1, 0))); # row vectors
      dr = tf.linalg.matmul(orthogonalized, residual);
      ur = tf.linalg.matmul(self.eigvec, centered, transpose_a = True, transpose_b = True);
      ul = tf.linalg.diag(self.sigma);
      dl = tf.zeros((dr.shape[0], ul.shape[1]), dtype = tf.float32);
      R = tf.concat([tf.concat([ul, ur], axis = 1),tf.concat([dl, dr], axis = 1)], axis = 0);
      s,u,v = tf.linalg.svd(R);
      self.eigvec = tf.linalg.matmul(tf.concat([self.eigvec, tf.transpose(orthogonalized, (1, 0))], axis = 1), u);
      self.sigma = s;
    return self.eigvec, self.sigma, self.mean;

if __name__ == "__main__":

  assert tf.executing_eagerly();
  A = tf.constant(np.random.normal(size = (10,128)), dtype = tf.float32);
  B = tf.constant(np.random.normal(size = (15,128)), dtype = tf.float32);
  skl = SKL(128);
  eigvec, sigma, mean = skl(A);
  print(eigvec, sigma, mean);
  eigvec, sigma, mean = skl(B);
  print(eigvec, sigma, mean);
