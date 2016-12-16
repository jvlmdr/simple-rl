import numpy as np
import gym
import tensorflow as tf

def train(env, create_policy, state_dim, action_dim, num_iters=1000, num_episodes=16, lr=1e-3, max_time_steps=1000):
    '''
    Parameters:
    create_policy -- Function that maps state to (logits_op, theta).
    '''

    history = {}
    history['reward'] = []
    history['num_episodes'] = []
    total_num_episodes = 0

    g = tf.Graph()
    with g.as_default():
        # Use variable batch size for computation of gradient.
        state_var = tf.placeholder(tf.float32, shape=(None, state_dim), name='state')
        logits_op, theta = create_policy(state_var)
        probs_op = tf.nn.softmax(logits_op)

        # Variables for computing loss (and gradient).
        action_var = tf.placeholder(tf.int32, shape=(None,), name='action')
        reward_var = tf.placeholder(tf.float32, shape=(None,), name='reward')
        loss_op = expected_reward(logits_op, action_var, reward_var, num_episodes)

        init_op = tf.global_variables_initializer()
        opt = tf.train.GradientDescentOptimizer(lr)
        train_op = opt.minimize(loss_op)

        sess = tf.Session()
        sess.run(init_op)

        def probs_fn(s):
            feed_dict = {state_var: np.reshape(s, (1, state_dim))} # Create batch.
            _, p = sess.run([logits_op, probs_op], feed_dict=feed_dict)
            return p[0]# Unpack from batch.

        for it in xrange(num_iters):
            # Construct a batch of inputs.
            states, actions, rewards = [], [], []
            for ep in xrange(num_episodes):
                s, a, r = run_episode(env, probs_fn, max_time_steps=max_time_steps)
                states.append(s)
                actions.append(a)
                rewards.append(r)
                total_num_episodes += 1
            episode_rewards = [sum(l) for l in rewards]
            feed_dict = {
                state_var:  np.array(concat(states)),
                action_var: np.array(concat(actions)),
                reward_var: np.array([episode_rewards[i] for i in range(num_episodes)
                                                         for j in range(len(rewards[i]))]),
            }
            _, _, loss = sess.run(
                [logits_op, train_op, loss_op],
                feed_dict=feed_dict)
            history['reward'].append(np.mean(episode_rewards))
            history['num_episodes'].append(total_num_episodes)
            print '%d  reward:%10.3e  loss:%10.3e' % (it, np.mean(episode_rewards), loss)

    return history

def concat(x):
    return [e for l in x for e in l]

def run_episode(env, policy, render=False, max_time_steps=1000):
    s, a, r = [], [], []
    s_t = env.reset()
    for t in xrange(max_time_steps):
        s.append(s_t)
        # Evaluate the neural network.
        probs = policy(s_t)
        # Sample the action.
        a_t = np.random.choice(len(probs), p=probs)
        a.append(a_t)
        s_t, r_t, done, info = env.step(a_t)
        r.append(r_t)
        if render:
            env.render()
        if done:
            break
    return s, a, r

def expected_reward(logits, actions, episode_rewards, num_episodes):
    actions = tf.to_int32(actions)
    # Get the log of the normalized logits for each action.
    # log(exp(logit[action]) / sum(exp(logit)))
    log_p_action = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, actions)
    return (1.0/num_episodes) * tf.reduce_sum(tf.mul(episode_rewards, log_p_action))
