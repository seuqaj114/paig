import numpy as np
import tensorflow as tf

""" Useful subnetwork components """


def unet(inp, base_channels, out_channels, upsamp=True):
    h = inp
    h = tf.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")
    h1 = tf.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")
    h = tf.layers.max_pooling2d(h1, 2, 2)
    h = tf.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")
    h2 = tf.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")
    h = tf.layers.max_pooling2d(h2, 2, 2)
    h = tf.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")
    h3 = tf.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")
    h = tf.layers.max_pooling2d(h3, 2, 2)
    h = tf.layers.conv2d(h, base_channels*8, 3, activation=tf.nn.relu, padding="SAME")
    h4 = tf.layers.conv2d(h, base_channels*8, 3, activation=tf.nn.relu, padding="SAME")
    if upsamp:
        h = tf.image.resize_bilinear(h4, h3.get_shape()[1:3])
        h = tf.layers.conv2d(h, base_channels*2, 3, activation=None, padding="SAME")
    else:
        h = tf.layers.conv2d_transpose(h, base_channels*4, 3, 2, activation=None, padding="SAME")
    h = tf.concat([h, h3], axis=-1)
    h = tf.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")
    h = tf.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")
    if upsamp:
        h = tf.image.resize_bilinear(h, h2.get_shape()[1:3])
        h = tf.layers.conv2d(h, base_channels*2, 3, activation=None, padding="SAME")
    else:
        h = tf.layers.conv2d_transpose(h, base_channels*2, 3, 2, activation=None, padding="SAME")
    h = tf.concat([h, h2], axis=-1)
    h = tf.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")
    h = tf.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")
    if upsamp:
        h = tf.image.resize_bilinear(h, h1.get_shape()[1:3])
        h = tf.layers.conv2d(h, base_channels*2, 3, activation=None, padding="SAME")
    else:
        h = tf.layers.conv2d_transpose(h, base_channels, 3, 2, activation=None, padding="SAME")
    h = tf.concat([h, h1], axis=-1)
    h = tf.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")
    h = tf.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")

    h = tf.layers.conv2d(h, out_channels, 1, activation=None, padding="SAME")
    return h


def shallow_unet(inp, base_channels, out_channels, upsamp=True):
    h = inp
    h = tf.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")
    h1 = tf.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")
    h = tf.layers.max_pooling2d(h1, 2, 2)
    h = tf.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")
    h2 = tf.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")
    h = tf.layers.max_pooling2d(h2, 2, 2)
    h = tf.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")
    h = tf.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")
    #h = tf.concat([h, h3], axis=-1)
    #h = tf.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")
    #h = tf.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")
    if upsamp:
        h = tf.image.resize_bilinear(h, h2.get_shape()[1:3])
        h = tf.layers.conv2d(h, base_channels*2, 3, activation=None, padding="SAME")
    else:
        h = tf.layers.conv2d_transpose(h, base_channels*2, 3, 2, activation=None, padding="SAME")
    h = tf.concat([h, h2], axis=-1)
    h = tf.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")
    h = tf.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")
    if upsamp:
        h = tf.image.resize_bilinear(h, h1.get_shape()[1:3])
        h = tf.layers.conv2d(h, base_channels*2, 3, activation=None, padding="SAME")
    else:
        h = tf.layers.conv2d_transpose(h, base_channels, 3, 2, activation=None, padding="SAME")
    h = tf.concat([h, h1], axis=-1)
    h = tf.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")
    h = tf.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")

    h = tf.layers.conv2d(h, out_channels, 1, activation=None, padding="SAME")
    return h


def variable_from_network(shape):
    # Produces a variable from a vector of 1's. 
    # Improves learning speed of contents and masks.
    var = tf.ones([1,10])
    var = tf.layers.dense(var, 200, activation=tf.tanh)
    var = tf.layers.dense(var, np.prod(shape), activation=None)
    var = tf.reshape(var, shape)
    return var
