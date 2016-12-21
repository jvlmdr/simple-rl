import tensorflow as tf

STATE_DIM = 4
ACTION_DIM = 2

def create_policy(x, use_value=False, h1_dim=10, h2_dim=10, h_stddev=1, action_stddev=0, value_stddev=1):
    with tf.name_scope('h1'):
        w1 = tf.Variable(tf.truncated_normal([STATE_DIM, h1_dim], stddev=h_stddev),
                         name='weight')
        b1 = tf.Variable(tf.zeros([h1_dim]), name='bias')
        h1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    with tf.name_scope('h2'):
        w2 = tf.Variable(tf.truncated_normal([h1_dim, h2_dim], stddev=h_stddev),
                         name='weight')
        b2 = tf.Variable(tf.zeros([h2_dim]), name='bias')
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

    with tf.name_scope('action'):
        w_action = tf.Variable(tf.truncated_normal([h2_dim, ACTION_DIM], stddev=action_stddev),
                               name='weight')
        b_action = tf.Variable(tf.zeros([ACTION_DIM]), name='bias')
        action = tf.matmul(h2, w_action) + b_action

    theta = [w1, b1, w2, b2, w_action, b_action]

    value = None
    if use_value:
        with tf.name_scope('value'):
            w_value = tf.Variable(tf.truncated_normal([h2_dim, 1], stddev=value_stddev),
                                  name='weight')
            b_value = tf.Variable(tf.zeros([1]), name='bias')
            value = tf.matmul(h2, w_value) + b_value
        theta.extend([w_value, b_value])

    return action, value, theta
