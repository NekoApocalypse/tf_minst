"""Build the MINST network
Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required to running the network foward to make the predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. trainint() -Adds to the loss mdel the Ops required to generate and apply gradients.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

NUM_CLASSES = 10
IMG_SIZE = 28
IMG_PIXELS = IMG_SIZE**2

def inference(images, hidden1_units, hidden2_units):
    """The MINST model to the point it may be used for inference
    Args:
        images: Images placeholder, from imputs()
        hidden1_units: Size of the first hidden layer
        hidden2_units: Size of the second hidden layer

    Returns:
        softmax_linear: Output tensor with the computed logits.
    """
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
                tf.truncated_normal([IMG_PIXELS, hidden1_units],
                    stddev=1.0 / math.sqrt(float(IMG_PIXELS))),
                name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),
                name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images,weights)+biases)
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
                tr.truncated_normal([hidden1_units,hidden2_units],
                    stddev=1.0 / math.sqrt(float(hidden1_units))),
                name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                name='biases')
        logits = tf.matmul(hidden2,weights) + biases
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
                tf.truncated_normal([hidden2_units,NUM_CLASSES],
                    stddev=1.0 / math.sqrt(float(hidden2_units))),
                name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                name='biases')
        logits = tf.matmul(hidden2,weights)+biases
        return logits

def loss(logits, labels):
    """Calculates the loss from the logits and the labels,
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size].

    Returns:
        loss: Loss tensor pf type float.
    """
    
    labels = tf.to_int64(labels)
    cross_entropy = tr.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def training(loss, learning_rate):
    """Sets up the training OPs
    
    Creates a summarizer to track the loss over time in TensorBoard.
    
    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the 'sess.run()' call to train the model.

    Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.

    Returns:
        train_op: The Op for training.
    """
    #Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss',loss)
    #Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #Create a variable to track the global step.
    global_step = tf.Variable(0,name='global_step',trainable=False)
    # Use the optimizer to apply the gradients to minimize the loss
    train_op = optimizer.minimize(loss,global_step=global_step)
    return train_op

def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    
    Args:
        logits: Logits tensor, flowt - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 = [batch_size]
    
    Returns:
        A scalar int32 tensor /w the number of examples that were prediected correctly.
    """
    # in_top_k(predictions, labels, k) op: return if the label is in top k predictions.
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32)


