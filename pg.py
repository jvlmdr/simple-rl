import numpy as np
import gym
import tensorflow as tf

ENV_NAME = 'CartPole-v0'
STATE_DIM = 4
ACTION_DIM = 2

def main():
    num_iters = 1000000
    num_episodes = 100
    max_time_steps = 1000
    learning_rate = 1e-3

    with tf.Graph().as_default():
        state_var = tf.placeholder(tf.float32, shape=(None, STATE_DIM), name="state")
        action_var = tf.placeholder(tf.int32, shape=(None,), name="action")
        reward_var = tf.placeholder(tf.float32, shape=(None,), name="reward")

        # Define operations.
        logits_op, weights1, weights2, weights3 = score_actions(state_var, 10, 10)
        probs_op = action_distribution(logits_op)
        init_op = tf.initialize_all_variables()
        loss_op = expected_reward(logits_op, action_var, reward_var, num_episodes)
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = opt.minimize(loss_op)
        grads_op = opt.compute_gradients(loss_op, (weights1, weights2, weights3))

        sess = tf.Session()
        sess.run(init_op)

        env = gym.make(ENV_NAME)
        for it in xrange(num_iters):
            states, actions, rewards = [], [], []
            episode_rewards = []
            for ep in xrange(num_episodes):
                st = env.reset()
                duration = 0
                r = 0.0
                for t in xrange(max_time_steps):
                    states.append(st)
                    # Evaluate the neural network.
                    feed_dict = {state_var: np.reshape(st, (1, STATE_DIM))}
                    _, probs = sess.run([logits_op, probs_op], feed_dict=feed_dict)
                    probs = probs[0]
                    # Sample the action.
                    at = np.random.choice(len(probs), p=probs)
                    actions.append(at)
                    st, rt, done, info = env.step(at)
                    r += rt
                    duration += 1
                    if done:
                        break
                # Provide the episode reward for every time-step.
                rewards.extend([r for _ in range(duration)])
                episode_rewards.append(r)
            # Construct a batch of inputs.
            feed_dict = {
                state_var:  np.array(states),
                action_var: np.array(actions),
                reward_var: np.array(rewards),
            }
            _, _, loss, grads = sess.run(
                [logits_op, train_op, loss_op, grads_op],
                feed_dict=feed_dict)
            print "reward: %.3g  loss: %.3g grads: %r" % (
                np.mean(episode_rewards),
                loss,
                [np.linalg.norm(g[0]) for g in grads])

def score_actions(obs, hidden1_dim, hidden2_dim):
    with tf.name_scope('hidden1'):
        weights1 = tf.Variable(
            tf.truncated_normal([STATE_DIM, hidden1_dim]),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_dim]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(obs, weights1) + biases)

    with tf.name_scope('hidden2'):
        weights2 = tf.Variable(
            tf.truncated_normal([hidden1_dim, hidden2_dim]),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden2_dim]), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights2) + biases)

    with tf.name_scope('output'):
        weights3 = tf.Variable(
            tf.truncated_normal([hidden2_dim, ACTION_DIM]),
            name='weights')
        biases = tf.Variable(tf.zeros([ACTION_DIM]), name='biases')
        logits = tf.matmul(hidden2, weights3) + biases

    return logits, weights1, weights2, weights3

def action_distribution(logits):
    return tf.nn.softmax(logits)

def expected_reward(logits, actions, rewards, num_episodes):
    actions = tf.to_int32(actions)
    return (1.0 / num_episodes) * tf.reduce_sum(tf.mul(
        rewards,
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits, actions)))

if __name__ == '__main__':
    main()
