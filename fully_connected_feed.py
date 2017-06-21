"""Trains and Evaluates the MNIST network using a feed dictionary"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time

from six.moves import xrange
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

FLAGS = None

def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
    To be used as inputs by the rest of the model, fed from downloaded data in the .run() loop.

    Args:
        batch_size: The batch size will be baked into both placeholders.

    Returns:
        images_placeholder: Images placeholder.
        labels_placeholder: Labels placeholder.
    """
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, images_pl, labels_pl):
    """Fills the feed_dict for training the given step.

    Arges:
        data_set: The set of images and labels, from input_dat.read_data_sets()
        images_pl: Images placeholder
        labes_pl: Labels placeholder

    Returns:
        feed:dict: The feed dictionary mapping from placeholders to vales.
    """
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
    feed_dict = {
            images_pl: images_feed,
            labels_pl: labesl_feed,
    }
    return feed_dict
