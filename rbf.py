# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 19:51:11 2018

@author: AwesomeCharacter
"""

import numpy as np
import tensorflow as tf

#helper function
def weight_variable(shape, weightName):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name = weightName)

def bias_variable(shape, biasName):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name = biasName)


def rbfFilters(clusterCenterArray, nodes):

  with tf.variable_scope("RBF") as sc:
    print ("nodes ", nodes)
    tensor = tf.constant(clusterCenterArray)
    tensor = tf.transpose(tensor)
    tensor = tf.expand_dims(tensor, 0)
    tensor = tf.expand_dims(tensor, 0)
    print (tensor)

  return tensor

clusters = 5 #odd number * 3
steps = 1/clusters

clustersCenterArray = []
for x in range(clusters+1):
    x = (-1 + (steps * (x) * 2 ))
    for y in range(clusters + 1):
        y = (-1 + (steps * (y) *2))
        for z in range (clusters +1):
            z = (-1 + (steps * (z) * 2 ))
            clusterCenter = [x,y,z]
            print (clusterCenter)
            clustersCenterArray.append(clusterCenter)
      

print (len(clustersCenterArray))

with tf.Graph().as_default():
    with tf.Session() as sess: 
        init_op = tf.global_variables_initializer()
#        rbfFilters(clustersCenterArray, len(clustersCenterArray))
        print (sess.run(rbfFilters(clustersCenterArray, len(clustersCenterArray))))

        
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        
        