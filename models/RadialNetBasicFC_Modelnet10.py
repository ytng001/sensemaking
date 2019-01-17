import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
#from transform_Vector import input_transform_Vector, inputvectorFeature
#from Transform_RBF_Feature import input_rbfTransform,feature_transform_net,input_transform_net
from RBF_InceptionNet import input_rbfTransform, input_RBFbasicLayer
def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    
    print ("RBF FC")
    with tf.variable_scope('RBFbasicLayer') as sc:
        net = input_RBFbasicLayer(point_cloud, is_training, bn_decay)
    
   
    net = tf_util.fully_connected(net, 1000, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
   

    
    
    print ("Net 1 ", net)
    net = tf_util.fully_connected(net, 1500, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    
    net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training,
                          scope='dp1')
         
    print ("Net 1 ", net)
    net = tf_util.fully_connected(net, 1000, bn=True, is_training=is_training,
                                  scope='fc3', bn_decay=bn_decay)
      

    print ("Net 2 ", net)
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc4', bn_decay=bn_decay)
#    
#    print ("Net 3 ", net)
#    net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training,
#                          scope='dp2')
    
    print ("Dopout 1 ", net)
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc5', bn_decay=bn_decay)
    print ("Net 4 ", net)
    net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training,
                          scope='dp3')
    
    print ("Dopout 2 ", net)
    net = tf_util.fully_connected(net, 10, activation_fn=None, scope='fc6')

    print ("Last ", net)
    return net, end_points


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    return classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
