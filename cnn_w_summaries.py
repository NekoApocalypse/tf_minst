import argparse
import sys
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def train():
    def weight_variable(shape):
        init = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init)

    def bias_variable(shape):
        init = tf.constant(0.1, shape=shape)
        return tf.Variable(init)

    def conv2d(x,w):
        return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    def variable_summaries(var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.histogram('histogram', var)

    def cnn_layer(input_tensor, layer_name):
        with tf.name_scope(layer_name):
            with tf.name_scope('conv1'):
                weights = weight_variable([5,5,1,32])
                biases = bias_variable([32])
                relu1 = tf.nn.relu(conv2d(input_tensor,weights)+biases)
            with tf.name_scope('pool1'):
                pool1 = max_pool_2x2(relu1)
            with tf.name_scope('conv2'):
                weights = weight_variable([5,5,32,64])
                biases = bias_variable([64])
                relu2 = tf.nn.relu(conv2d(pool1,weights)+biases)
            with tf.name_scope('pool2'):
                pool2 = max_pool_2x2(relu2)
            with tf.name_scope('FC1'):
                weights = weight_variable([7*7*64,1024])
                biases = bias_variable([1024])
                pool2_flat = tf.reshape(pool2,[-1,7*7*64])
                relu3 = tf.nn.relu(tf.matmul(pool2_flat,weights)+biases)
            with tf.name_scope('dropout'):
                keep_prob = tf.placeholder(tf.float32)
                drop = tf.nn.dropout(relu3, keep_prob)
            with tf.name_scope('FC2'):
                weights = weight_variable([1024,10])
                biases = bias_variable([10])
                predict = tf.matmul(drop,weights)+biases
        return predict, keep_prob

    mnist = input_data.read_data_sets(FLAGS.data_dir,
                                      one_hot=True,
                                      fake_data=FLAGS.fake_data)
    sess = tf.Session()
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
    with tf.name_scope('reshape'):
        x_reshape = tf.reshape(x,[-1,28,28,1])
        tf.summary.image('input',x_reshape,10)

    y,keep_prob = cnn_layer(x_reshape,'CNN_Layer')
    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', cross_entropy)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
            cross_entropy)
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy',accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train/', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    sess.run(tf.global_variables_initializer())

    def feed_dict(train):
        if train or FLAGS.fake_data:
            xs,ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
            k = FLAGS.dropout
        else:
            xs,ys = mnist.test.images, mnist.test.labels
            k = 1.0
        return {x:xs, y_:ys, keep_prob:k}

    for i in range(FLAGS.max_steps):
        if i% 10 ==0:
            summary,acc = sess.run([merged, accuracy],
                                   feed_dict=feed_dict(False))
            test_writer.add_summary(summary,i)
            print('Accuracy at step %s: %s' % (i,acc))
        else:
            if i%100 == 99:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged,train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata,'step%03d'%i)
                train_writer.add_summary(summary,i)
                print('Adding run metadata for', i)
            else:
                summary,_ = sess.run([merged,train_step],
                                     feed_dict=feed_dict(True))
                train_writer.add_summary(summary,i)
    train_writer.close()
    test_writer.close()

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                        default=False,
                        help='If true, uses fake data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='Keep probability for training dropout.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./tmp/input_data',
        help='Directory for storing input data')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./tmp/logs',
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)









