#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;

@tf.function
def gram_schmidt(inputs):
  # inputs.shape = (number of eigenvector, dimension of eigenvector)
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

def SKL():
  pass

if __name__ == "__main__":

  assert tf.executing_eagerly();
  a = tf.constant(np.random.normal(size = (10,128)), dtype = tf.float32);
  b = gram_schmidt(a);
  print(gram_schmidt);
