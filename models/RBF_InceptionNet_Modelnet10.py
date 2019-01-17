import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from sklearn.metrics.pairwise import rbf_kernel

clusters = 9 #odd number * 3
steps = 1/clusters

#define cluster centrod

def input_RBFbasicLayer(point_cloud, is_training, bn_decay=None):

    print ("point_cloud ", point_cloud)
    batch_size = point_cloud.get_shape()[0].value
    print ("batch_size ", batch_size)
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
    print ("Cluster centroids" ,len(clustersCenterArray)) 
    
        
    with tf.variable_scope("RBF_BasicLayer") as sc:      
        tensor = tf.constant(clustersCenterArray)

        input_reshape = tf.reshape(point_cloud, [-1, 3])
          
        exp_Input = tf.expand_dims(input_reshape, 1)
        exp_Clusters = tf.expand_dims(tensor, 0)
        
        print ("input_reshape ", exp_Input)
        print ("tensor ", exp_Clusters)
        sigma =0.15
        distanceSquare = tf.reduce_sum(tf.squared_difference(exp_Input, exp_Clusters),2)
        rbfInput = tf.exp(-distanceSquare / (2* sigma)) #assuming sigma is 1.0

        print ("rbf Input ", rbfInput)
        #Add conv and MLP layer here
        print ("Num of points ", num_point)
        net = tf.reshape(rbfInput, [batch_size,num_point ,len(clustersCenterArray)])
        
        collapsedInput = tf.reduce_max(net,1,keepdims = True)
        collapsedInput = tf.reshape(collapsedInput, [batch_size ,len(clustersCenterArray)])
        print ("Reshape Net ", collapsedInput)
    return collapsedInput

def input_InceptionNet(point_cloud, is_training, bn_decay=None, K=3):

    print ("input_InceptionNet" , point_cloud)
#    numOfClusters = point_cloud.get_shape()[2].value
    
    input_reshape = tf.reshape(point_cloud, [32,1,1,512])
    
    print ("input_reshape" , input_reshape)
    net = tf_util.conv2d(input_reshape, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='rbf_fc1', bn_decay=bn_decay)

    net = tf_util.conv2d(net, 256, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='rbf_fc6', bn_decay=bn_decay)  
        
    net = tf_util.conv2d(net, 1024, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='rbf_fc7', bn_decay=bn_decay)  
      


    net = tf.reshape(net, [32, -1])
    print ("last net ", net) 
    return net

def input_RBF_BasicTransform(point_cloud, is_training, bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    print ("point_cloud ", point_cloud)
    batch_size = point_cloud.get_shape()[0].value
    print ("batch_size ", batch_size)
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
    print ("Cluster centroids" ,len(clustersCenterArray)) 
    
        
    with tf.variable_scope("RBF_fullyConnected") as sc:      
        tensor = tf.constant(clustersCenterArray)

        input_reshape = tf.reshape(point_cloud, [-1, 3])
          
        exp_Input = tf.expand_dims(input_reshape, 1)
        exp_Clusters = tf.expand_dims(tensor, 0)
        
        print ("input_reshape ", exp_Input)
        print ("tensor ", exp_Clusters)
        sigma =0.15
        distanceSquare = tf.reduce_sum(tf.squared_difference(exp_Input, exp_Clusters),2)
        rbfInput = tf.exp(-distanceSquare / (2* sigma)) #assuming sigma is 1.0
        
        net = tf.reshape(rbfInput, [batch_size,num_point ,len(clustersCenterArray)])
        
        collapsedInput = tf.reduce_max(net,1,keepdims = True)
        collapsedInput = tf.reshape(collapsedInput, [batch_size ,len(clustersCenterArray)])
    
       
    return collapsedInput
 
def input_rbfTransform(point_cloud, is_training, bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    print ("point_cloud ", point_cloud)
    batch_size = point_cloud.get_shape()[0].value
    print ("batch_size ", batch_size)
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
    print ("Cluster centroids" ,len(clustersCenterArray)) 
    
        
    with tf.variable_scope("RBF") as sc:      
        tensor = tf.constant(clustersCenterArray)

        input_reshape = tf.reshape(point_cloud, [-1, 3])
          
        exp_Input = tf.expand_dims(input_reshape, 1)
        exp_Clusters = tf.expand_dims(tensor, 0)
        
        print ("input_reshape ", exp_Input)
        print ("tensor ", exp_Clusters)
        sigma =0.2
        distanceSquare = tf.reduce_sum(tf.squared_difference(exp_Input, exp_Clusters),2)
        rbfInput = tf.exp(-distanceSquare / (2* sigma)) #assuming sigma is 1.0


        #Add conv and MLP layer here
        input_reshape = tf.reshape(rbfInput, [batch_size,num_point,-1 ,len(clustersCenterArray)])
        net = tf_util.conv2d(input_reshape, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='rbf_fc1', bn_decay=bn_decay)
        print ("convl ", net)
        net = tf_util.conv2d(net, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='rbf_fc2', bn_decay=bn_decay)
        
        net = tf_util.conv2d(net, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='rbf_fc3', bn_decay=bn_decay)  
                
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='rbf_fc4', bn_decay=bn_decay)  
        
        net = tf_util.conv2d(net, 1024, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='rbf_fc5', bn_decay=bn_decay)  
      
        net = tf_util.max_pool2d(net, [num_point,1],
                                 padding='VALID', scope='tmaxpool1')
        
        print ("last net ", net)
        net = tf.reshape(net, [batch_size, -1])
       
    return net

def input_transform_net(point_cloud, is_training, bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    input_image = tf.expand_dims(point_cloud, -1)
    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
#    net = tf_util.conv2d(net, 128, [1,1],
#                         padding='VALID', stride=[1,1],
#                         bn=True, is_training=is_training,
#                         scope='tconv2', bn_decay=bn_decay)
#    net = tf_util.conv2d(net, 1024, [1,1],
#                         padding='VALID', stride=[1,1],
#                         bn=True, is_training=is_training,
#                         scope='tconv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    
#    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
#                          scope='dp1')
#        
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
    return transform







