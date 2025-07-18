import math
import numpy as np 
import tensorflow as tf

# Helper for batch norm
BN = tf.keras.layers.BatchNormalization

# Helper for variable creation

def get_weight(shape, stddev=0.01, name=None):
    return tf.Variable(tf.random.truncated_normal(shape, stddev=stddev), name=name)

def get_bias(shape, bias_start=0.0, name=None):
    return tf.Variable(tf.zeros(shape) + bias_start, name=name)

def conv2d(input_, output, kernel, stride, relu, bn, name, stddev=0.01):
    shape = [kernel, kernel, input_.shape[-1], output]
    w = get_weight(shape, stddev=stddev, name=f'{name}_w')
    conv = tf.nn.conv2d(input_, w, strides=[1, stride, stride, 1], padding='SAME')
    b = get_bias([output], name=f'{name}_b')
    conv = tf.nn.bias_add(conv, b)
    if bn:
        conv = BN()(conv)
    if relu:
        conv = tf.nn.relu(conv, name=name)
    return conv

def max_pool(input_, kernel, stride, name):
    return tf.nn.max_pool(input_, ksize=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)

def linear(input_, output, name, stddev=0.02, bias_start=0.0):
    shape = input_.shape.as_list()
    matrix = get_weight([shape[1], output], stddev=stddev, name=f'{name}_Matrix')
    bias = get_bias([output], bias_start=bias_start, name=f'{name}_bias')
    return tf.matmul(input_, matrix) + bias

def atrous_conv2d(input_, output, kernel, rate, relu, name, stddev=0.01):
    shape = [kernel, kernel, input_.shape[-1], output]
    w = get_weight(shape, stddev=stddev, name=f'{name}_w')
    conv = tf.nn.atrous_conv2d(input_, w, rate, padding='SAME')
    b = get_bias([output], name=f'{name}_b')
    conv = tf.nn.bias_add(conv, b)
    if relu:
        conv = tf.nn.relu(conv, name=name)
    return conv

def gcn(input_, output, kernel, stride, relu, bn, name, stddev=0.01):
    left_shape_k_1 = [kernel, 1, input_.shape[-1], output]
    left_shape_1_k = [1, kernel, output, output]
    right_shape_1_k = [1, kernel, input_.shape[-1], output]
    right_shape_k_1 = [kernel, 1, output, output]
    w1_1 = get_weight(left_shape_k_1, stddev=stddev, name=f'{name}_w1_1')
    w1_2 = get_weight(left_shape_1_k, stddev=stddev, name=f'{name}_w1_2')
    w2_1 = get_weight(right_shape_1_k, stddev=stddev, name=f'{name}_w2_1')
    w2_2 = get_weight(right_shape_k_1, stddev=stddev, name=f'{name}_w2_2')
    b1_1 = get_bias([output], name=f'{name}_b1_1')
    b1_2 = get_bias([output], name=f'{name}_b1_2')
    b2_1 = get_bias([output], name=f'{name}_b2_1')
    b2_2 = get_bias([output], name=f'{name}_b2_2')

    conv1_1 = tf.nn.conv2d(input_, w1_1, strides=[1, stride, stride, 1], padding='SAME')
    conv1_1 = tf.nn.bias_add(conv1_1, b1_1)
    if bn:
        conv1_1 = BN()(conv1_1)
    if relu:
        conv1_1 = tf.nn.relu(conv1_1, name=f'{name}_conv1_1')

    conv1_2 = tf.nn.conv2d(conv1_1, w1_2, strides=[1, stride, stride, 1], padding='SAME')
    conv1_2 = tf.nn.bias_add(conv1_2, b1_2)
    if bn:
        conv1_2 = BN()(conv1_2)
    if relu:
        conv1_2 = tf.nn.relu(conv1_2, name=f'{name}_conv1_2')
    
    conv2_1 = tf.nn.conv2d(input_, w2_1, strides=[1, stride, stride, 1], padding='SAME')
    conv2_1 = tf.nn.bias_add(conv2_1, b2_1)
    if bn:
        conv2_1 = BN()(conv2_1)
    if relu:
        conv2_1 = tf.nn.relu(conv2_1, name=f'{name}_conv2_1')

    conv2_2 = tf.nn.conv2d(conv2_1, w2_2, strides=[1, stride, stride, 1], padding='SAME')
    conv2_2 = tf.nn.bias_add(conv2_2, b2_2)
    if bn:
        conv2_2 = BN()(conv2_2)
    if relu:
        conv2_2 = tf.nn.relu(conv2_2, name=f'{name}_conv2_2')

    top = tf.add_n([conv1_2, conv2_2])
    return top

def br(input_, output, kernel, stride, name):
    br_conv1 = conv2d(input_, output, kernel, stride, relu=True, bn=False, name=f'{name}_br_conv1')
    br_conv2 = conv2d(br_conv1, output, kernel, stride, relu=False, bn=False, name=f'{name}_br_conv2')
    top = tf.add_n([input_, br_conv2])
    return top

def residual_module(input_, output, is_BN, name):
    mid_channel = output >> 1
    conv1 = conv2d(input_, mid_channel, 1, 1, relu=True, bn=is_BN, name=f'{name}_res_conv1')
    conv2 = conv2d(conv1, mid_channel, 3, 1, relu=True, bn=is_BN, name=f'{name}_res_conv2')
    conv3 = conv2d(conv2, output, 1, 1, relu=False, bn=is_BN, name=f'{name}_res_conv3')
    conv_side = conv2d(input_, output, 1, 1, relu=False, bn=is_BN, name=f'{name}_res_conv_side')
    top = tf.add_n([conv3, conv_side])
    top = tf.nn.relu(top, name=f'{name}_res_relu')
    return top 

def gcn_residual_module(input_, output, gcn_kernel, is_BN, name):
    mid_channel = output >> 1
    gcn_layer = gcn(input_, mid_channel, gcn_kernel, 1, relu=True, bn=is_BN, name=f'{name}_gcn_residual1')
    conv1 = conv2d(gcn_layer, output, 1, 1, relu=False, bn=is_BN, name=f'{name}_gcn_residual2')
    conv_side = conv2d(input_, output, 1, 1, relu=False, bn=is_BN, name=f'{name}_gcn_residual3')
    top = tf.add_n([conv1, conv_side])
    top = tf.nn.relu(top, name=f'{name}_gcn_res_relu')
    return top