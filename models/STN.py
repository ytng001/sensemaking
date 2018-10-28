# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 01:39:46 2018

@author: ytng0
"""
import tensorflow as tf
import numpy as np

def grid_generator(hight, width, theta):
  x_t = -1 + tf.range(width) / width * 2
  x_t = tf.reshape(tf.concat([x_t] * hight, axis = 0), \
                     [1, hight * width])
  x_t = tf.cast(x_t, dtype = tf.float32)

  y_t = -1 + tf.range(hight) / hight * 2
  y_t = tf.tile(tf.reshape(y_t, [hight, 1]), [1, width])
  y_t = tf.reshape(y_t, [1, hight * width])
  y_t = tf.cast(y_t, dtype = tf.float32)

  ones_t = tf.ones(shape = [1, hight * width], \
                      dtype = tf.float32)

  grids_t = tf.concat([x_t, y_t, ones_t], axis = 0)

  A = tf.concat([theta, [[0, 0, 1]]], axis = 0)

  grids_s = tf.matmul(A, grids_t)
  grids_s = tf.concat([[grids_s[0]], [grids_s[1]]], axis = 0)

  return grids_s

def sampler(img_in, grids, hight, width):
  x = (1.0 + grids[0]) * width / 2.0
  y = (1.0 + grids[1]) * hight / 2.0

  x0 = tf.cast(tf.floor(x), tf.int32) 
  x1 = x0 + 1
  y0 = tf.cast(tf.floor(y), tf.int32) 
  y1 = y0 + 1

  w_ul = (tf.cast(x1, tf.float32) - x) *   (tf.cast(y1, tf.float32) - y)
  w_ll = (tf.cast(x1, tf.float32) - x) *   (y - tf.cast(y0, tf.float32))
  w_ur = (x - tf.cast(x0, tf.float32)) * (tf.cast(y1, tf.float32) - y)
  w_lr = (x - tf.cast(x0, tf.float32)) * (y - tf.cast(y0, tf.float32))

  x0_img = tf.maximum(0, tf.minimum(x0, width - 1))
  x1_img = tf.maximum(0, tf.minimum(x1, width - 1))
  y0_img = tf.maximum(0, tf.minimum(y0, hight - 1))
  y1_img = tf.maximum(0, tf.minimum(y1, hight - 1))

  x0_img = tf.expand_dims(x0_img, axis = 1)
  x1_img = tf.expand_dims(x1_img, axis = 1)
  y0_img = tf.expand_dims(y0_img, axis = 1)
  y1_img = tf.expand_dims(y1_img, axis = 1)

  idx_ul = tf.concat([y0_img, x0_img], axis = 1)
  idx_ll = tf.concat([y1_img, x0_img], axis = 1)
  idx_ur = tf.concat([y0_img, x1_img], axis = 1)
  idx_lr = tf.concat([y1_img, x1_img], axis = 1)

  img_ul = tf.gather_nd(img_in, idx_ul)
  img_ll = tf.gather_nd(img_in, idx_ll)
  img_ur = tf.gather_nd(img_in, idx_ur)
  img_lr = tf.gather_nd(img_in, idx_lr)

  img_out = w_ul * img_ul + w_ll * img_ll +    w_ur * img_ur + w_lr * img_lr

  return img_out


# MNIST 
index = np.random.randint(1000)
img_in = np.reshape(mnist.train.images[index], [28, 28])

hight = 28
width = 28
theta = tf.constant([[1, 0, 0], [0, 1, 0]], \
                         dtype = tf.float32)

grids = grid_generator(hight, width, theta)
img_out = tf.reshape(sampler(img_in, grids, hight, width), \
                         [hight, width])

with tf.Session() as sess:

  img_out = sess.run(img_out)

  print ('theta')
  print (sess.run(theta))

#   
fig = plt.figure(figsize = (8, 6))

ax = fig.add_subplot(1, 2, 1)
ax.imshow(img_in, cmap = 'gray')
ax.set_title('Input')
ax.set_axis_off()

ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(img_out, cmap = 'gray')
ax2.set_title('Transformed')
ax2.set_axis_off()

plt.show()