# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 01:44:23 2018

@author: ytng0
"""

import tensorflow as tf
import numpy as np

def grid_generator(hight, width, theta):
  x_t = -1 + tf.range(width) / width * 2
  x_t = tf.reshape(tf.concat([x_t] * hight, axis = 0),   [1, hight * width])
  x_t = tf.cast(x_t, dtype = tf.float32)

  y_t = -1 + tf.range(hight) / hight * 2
  y_t = tf.tile(tf.reshape(y_t, [hight, 1]), [1, width])
  y_t = tf.reshape(y_t, [1, hight * width])
  y_t = tf.cast(y_t, dtype = tf.float32)

  ones_t = tf.ones(shape = [1, hight * width],  dtype = tf.float32)

  print (x_t)
  print (y_t)
  grids_t = tf.concat([x_t, y_t, ones_t], axis = 0)
  print (grids_t)
  
  
  A = tf.concat([theta, [[0, 0, 1]]], axis = 0)
  print ("A" , A)
  grids_s = tf.matmul(A, grids_t)
  
  print ("grids_s" ,grids_s)
  grids_s = tf.concat([[grids_s[0]], [grids_s[1]]], axis = 0)

  print ("grids_s" ,grids_s)
  return grids_s

theta = tf.constant([[1, 0, 0], [0, 1, 0]], dtype = tf.float32)
print (grid_generator(10,10,theta))