import tensorflow as tf

STATE_DIM = 4
ACTION_DIM = 2

def create_policy(obs, h1_dim=10, h2_dim=10, h_stddev=1, out_stddev=1):
    with tf.name_scope('h1'):
        w1 = tf.Variable(tf.truncated_normal([STATE_DIM, h1_dim], stddev=h_stddev),
                         name='weight')
        b1 = tf.Variable(tf.zeros([h1_dim]), name='bias')
        h1 = tf.nn.relu(tf.matmul(obs, w1) + b1)

    with tf.name_scope('h2'):
        w2 = tf.Variable(tf.truncated_normal([h1_dim, h2_dim], stddev=h_stddev),
                         name='weight')
        b2 = tf.Variable(tf.zeros([h2_dim]), name='bias')
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

    with tf.name_scope('output'):
        w3 = tf.Variable(tf.truncated_normal([h2_dim, ACTION_DIM], stddev=out_stddev),
                         name='weight')
        b3 = tf.Variable(tf.zeros([ACTION_DIM]), name='bias')
        logits = tf.matmul(h2, w3) + b3

    theta = (w1, b1, w2, b2, w3, b3)
    return logits, theta
