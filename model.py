import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils


def inference(inputs, batch_size, num_classes, training=True):
    with tf.variable_scope('inference') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            cnv1 = slim.conv2d(inputs, 16, [3, 3], stride=1, scope='cnv1')
            cnv2 = slim.conv2d(cnv1, 32, [1, 1], stride=1, scope='cnv2')
            max_pool1 = slim.max_pool2d(cnv2, [3, 3], stride=2, scope='maxpool1')
            cnv3 = slim.conv2d(max_pool1, 32, [3, 3], stride=1, scope='cnv3')
            cnv4 = slim.conv2d(cnv3, 64, [1, 1], stride=1, scope='cnv4')
            max_pool2 = slim.max_pool2d(cnv4, [3, 3], stride=2, scope='maxpool2')
            flat = slim.flatten(max_pool2, scope='flatten')
            fc_1 = slim.fully_connected(flat, 128, scope='fc_1')
            drop1 = slim.dropout(fc_1, scope='drop1')
            fc_2 = slim.fully_connected(drop1, num_classes, scope='fc_2')
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return fc_2, end_points_collection


def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                       name='cross_entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


def training(loss, learning_rate):
    with tf.variable_scope('train_steps') as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_steps = tf.Variable(0, trainable=False, name='global_steps')
        train_op = optimizer.minimize(loss, global_step=global_steps)
    return train_op


def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        predictions = tf.equal(tf.argmax(logits, 1), tf.cast(labels, tf.int64))
        acc = tf.reduce_mean(tf.cast(predictions, tf.float32), name='result')
        tf.summary.scalar(acc.name, acc)
    return acc
