import tensorflow as tf

def create_policy(x, input_dim, action_dim, use_value=False, h1_dim=10, h_stddev=1, action_stddev=0, value_stddev=1):
    with tf.name_scope('action'):
        w_action = tf.Variable(tf.truncated_normal([input_dim, action_dim], stddev=action_stddev),
                               name='weight')
        b_action = tf.Variable(tf.zeros([action_dim]), name='bias')
        action = tf.matmul(x, w_action) + b_action
    theta = [w_action, b_action]

    value = None
    if use_value:
        with tf.name_scope('h1'):
            w1 = tf.Variable(tf.truncated_normal([input_dim, h1_dim], stddev=h_stddev),
                             name='weight')
            b1 = tf.Variable(tf.zeros([h1_dim]), name='bias')
            h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
        with tf.name_scope('value'):
            w_value = tf.Variable(tf.truncated_normal([h1_dim, 1], stddev=value_stddev),
                                  name='weight')
            b_value = tf.Variable(tf.zeros([1]), name='bias')
            value = tf.matmul(h1, w_value) + b_value
        theta.extend([w1, b1, w_value, b_value])

    return action, value, theta
