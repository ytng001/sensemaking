import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from sklearn.metrics.pairwise import rbf_kernel

clusters = 11 #odd number * 3
steps = 1/clusters

#define cluster centrod

            
def input_rbfTransform(point_cloud, is_training, bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    print ("point_cloud ", point_cloud)
    batch_size = point_cloud.get_shape()[0].value
    print ("batch_size ", batch_size)
    num_point = point_cloud.get_shape()[1].value
    clustersCenterArray = []
    
    print ("Create cluster centroids")
    for x in range(clusters+1):
        x = (-1 + (steps * (x) * 2 ))
        for y in range(clusters + 1):
            y = (-1 + (steps * (y) *2))
            for z in range (clusters +1):
                z = (-1 + (steps * (z) * 2 ))
                clusterCenter = [x,y,z]
                clustersCenterArray.append(clusterCenter)
                
    with tf.variable_scope("RBF") as sc:
        newImage = tf.expand_dims(point_cloud, [2])
        print ("RBF", newImage)       
        tensor = tf.constant(clustersCenterArray)

        input_reshape = tf.reshape(point_cloud, [-1, 3])
          
        exp_Input = tf.expand_dims(input_reshape, 1)
        exp_Clusters = tf.expand_dims(tensor, 0)
        
        print ("input_reshape ", exp_Input)
        print ("tensor ", exp_Clusters)
        distanceSquare = tf.reduce_sum(tf.squared_difference(exp_Input, exp_Clusters),2)
        rbfInput = tf.exp(-distanceSquare) #assuming sigma is 1.0

        print ("rbfInput ", rbfInput)
    
    
    rbfInput = tf.reshape(rbfInput, [batch_size, -1, 1, len(clustersCenterArray)])
#    input_image = tf.expand_dims(point_cloud, -1)
#    
#    print ("input_image ", input_image)

    
    print ("end transform")
    return rbfInput


