import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from sklearn.metrics.pairwise import rbf_kernel

clusters = 13 #14#odd number * 3
steps = 1/clusters

#define cluster centrod

            
def input_rbfTransform(point_cloud, is_training, bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix o size 3xK """

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    clustersCenterArray = []
    

    for x in range(clusters+1):
        x = (-1 + (steps * (x) * 2 ))
        for y in range(clusters + 1):
            y = (-1 + (steps * (y) *2))
            for z in range (clusters +1):
                z = (-1 + (steps * (z) * 2 ))
                clusterCenter = [x,y,z]
                clustersCenterArray.append(clusterCenter)
    
    print ("total clusters ", len(clustersCenterArray))
    with tf.variable_scope("RBF") as sc:      
        tensor = tf.constant(clustersCenterArray)

        input_reshape = tf.reshape(point_cloud, [-1, 3])
          
        exp_Input = tf.expand_dims(input_reshape, 1)
        exp_Clusters = tf.expand_dims(tensor, 0)

#        print ("input_reshape ", exp_Input)
#        print ("tensor ", exp_Clusters()
        sigma = 0.6
#        print ("exp input ", exp_Input)
#        print ("exp_Clusters ", exp_Clusters)
#        squaredDiff = tf.squared_difference(exp_Input, exp_Clusters)
        distanceSquare = tf.reduce_sum(tf.squared_difference(exp_Input, exp_Clusters),2)

        rbfInput = tf.exp(-distanceSquare / (2* sigma * sigma)) #assuming sigma is 1.0
       
        mask =   tf.where(
        tf.equal(tf.reduce_max(rbfInput, axis=1, keep_dims=True), rbfInput), 
        tf.constant(1.0, shape=rbfInput.shape), 
        tf.constant(1.0, shape=rbfInput.shape)
        )
        
        rbfInput = tf.multiply(mask, rbfInput) #SEt non max value to 0
       
        
        #Add conv and MLP layer here
        
        tempReshape = tf.reshape(rbfInput, [batch_size, num_point ,len(clustersCenterArray)])
        tempReshape = tf.reduce_max(tempReshape, 1)
        out = tempReshape
        print ("length of cluster ",len(clustersCenterArray))
        input_reshape = tf.reshape(tempReshape, [batch_size,1,1 ,len(clustersCenterArray)])
    
        print ("rbfInput" , input_reshape)
       
#        net = tf_util.conv2d(input_reshape, 256, [1,1],
#                             padding='VALID', stride=[1,1],
#                             bn=True, is_training=is_training,
#                             scope='rbf_fc1', bn_decay=bn_decay)
#        
#        
#        net = tf_util.conv2d(net, 256, [1,1],
#                             padding='VALID', stride=[1,1],
#                             bn=True, is_training=is_training,
#                             scope='rbf_fc3', bn_decay=bn_decay)  
#        
#         
#        net = tf_util.conv2d(net, 512, [1,1],
#                             padding='VALID', stride=[1,1],
#                             bn=True, is_training=is_training,
#                             scope='rbf_fc4', bn_decay=bn_decay)  
# 
#
#        net = tf_util.conv2d(net, 1024, [1,1],
#                             padding='VALID', stride=[1,1],
#                             bn=True, is_training=is_training,
#                             scope='rbf_fc7', bn_decay=bn_decay)  
      
#        net = tf_util.max_pool2d(net, [num_point,1],
#                                 padding='VALID', scope='tmaxpool1')
#        
#        
#        print ("last net ", net)
        net = tf.reshape(input_reshape, [batch_size, -1])
        
#        print ("net reshape ",net)
#        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
#                                      scope='tfc1', bn_decay=bn_decay)
#
#        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
#                                      scope='tfc2', bn_decay=bn_decay)

       
    return net,out


def input_rbfFeatureVector(inputs, is_training, bn_decay=None, K=64):
    batch_size = inputs.get_shape()[0].value
    num_point = inputs.get_shape()[1].value

    clustersCenterArray = []
    

    for x in range(clusters+1):
        x = (-1 + (steps * (x) * 2 ))
        for y in range(clusters + 1):
            y = (-1 + (steps * (y) *2))
            for z in range (clusters +1):
                z = (-1 + (steps * (z) * 2 ))
                clusterCenter = [x,y,z]
                clustersCenterArray.append(clusterCenter)
    print ("Cluster centroids" ,len(clustersCenterArray))
    tensor = tf.constant(clustersCenterArray) 
    
    net = tf_util.conv2d(inputs, 3, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    
    input_reshape = tf.reshape(net, [-1, 3])
          
    exp_Input = tf.expand_dims(input_reshape, 1)
    exp_Clusters = tf.expand_dims(tensor, 0)
        
    distanceSquare = tf.reduce_sum(tf.squared_difference(exp_Input, exp_Clusters),2)
    rbfInput = tf.exp(-distanceSquare) #assuming sigma is 1.0


    input_reshape = tf.reshape(rbfInput, [batch_size,num_point,-1 ,len(clustersCenterArray)])
    
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









