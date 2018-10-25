import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from sklearn.metrics.pairwise import rbf_kernel



            
def input_transform_Vector(point_cloud, is_training, bn_decay=None, K=3):
    originCentroid = [0.0,0.0,0.0]
    clustersCenterArray = []
    clustersCenterArray.append(originCentroid)
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
            
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

                
    with tf.variable_scope("VectorTransform") as sc:
  
        tensor = tf.constant(clustersCenterArray)
        input_reshape = tf.reshape(point_cloud, [-1, 3])
          
        exp_Input = tf.expand_dims(input_reshape, 1)
        exp_Clusters = tf.expand_dims(tensor, 0)
        
        directionVector = tf.subtract(exp_Clusters, exp_Input)
        print ("directionVector ", directionVector)      
        directionSum = tf.reduce_sum(directionVector,2, keepdims = True)
        unitVector = tf.divide(directionVector,directionSum)
        
        unitVector = tf.reshape(unitVector,[-1, 3])
        distanceSquare = tf.reduce_sum(tf.squared_difference(exp_Input, exp_Clusters),2)     
        rbfInput = tf.exp(-distanceSquare) #assuming sigma is 1.0
        
        print ("unitVector", unitVector)
        print ("rbfInput ", rbfInput)
        concatInput = tf.concat([unitVector, rbfInput], 1)
#        rbfInput = tf.reshape(rbfInput, [batch_size, -1, 1, len(clustersCenterArray)])
        
        net = tf.reshape(concatInput, [batch_size, -1])
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                      scope='tfc1', bn_decay=bn_decay)
        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                      scope='tfc2', bn_decay=bn_decay)
    
        with tf.variable_scope('transform_XYZ') as sc:
            assert(K==3)
            weights = tf.get_variable('weights', [256, 3*K],
                                      initializer=tf.constant_initializer(0.0),
                                      dtype=tf.float32)
            biases = tf.get_variable('biases', [3*K],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=tf.float32)
            biases += tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
            transform = tf.matmul(net, weights)
            transform = tf.nn.bias_add(transform, biases)
    
        transform = tf.reshape(transform, [batch_size, 3, K])
    
        
    
    print ("end input_transform_Vector")
    return transform, concatInput





def inputvectorFeature(inputs, is_training, bn_decay=None, K=64):
    """ Feature Transform Net, input is BxNx1xK
        Return:
            Transformation matrix of size KxK """
    batch_size = inputs.get_shape()[0].value
    num_point = inputs.get_shape()[1].value

    net = tf_util.conv2d(inputs, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_feat') as sc:
        weights = tf.get_variable('weights', [256, K*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [K*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform

























