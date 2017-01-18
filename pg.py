import numpy as np
import gym
import tensorflow as tf

def train(env, create_policy, state_dim, action_dim, num_iters=1000, num_episodes=16, lr=1e-3, discount=1.0, max_time_steps=1000, use_advantage=False, coeff_value=1.0, weight_decay=0.0):
    '''
    Parameters:
    create_policy -- Function that maps state to (logits_op, theta).
    '''

    h = {
        'num_episodes': [],
        'reward':       [],
        # Record max-norm of gradients.
        'gradient': [],
        # Record future reward and value estimate.
        'mean_future_reward': [],
        'mean_value':         [],
    }
    total_num_episodes = 0

    g = tf.Graph()
    with g.as_default():
        # Use variable batch size for computation of gradient.
        state_var = tf.placeholder(tf.float32, shape=(None, state_dim), name='state')
        logits_op, value_op, theta = create_policy(state_var)
        probs_op = tf.nn.softmax(logits_op)

        # Variables for computing loss (and gradient).
        action_var = tf.placeholder(tf.int32, shape=(None,), name='action')
        # total_reward_var = tf.placeholder(tf.float32, shape=(None,), name='total_reward')
        future_reward_var = tf.placeholder(tf.float32, shape=(None,), name='future_reward')
        sample_weight_var = tf.placeholder(tf.float32, shape=(None,), name='sample_weight')

        if use_advantage:
            reward_loss_op = advantage_loss(logits_op, value_op, action_var, future_reward_var, sample_weight_var)
            value_loss_op = 0.5*tf.reduce_mean(tf.square(value_op - future_reward_var))
            loss_op = reward_loss_op + coeff_value*value_loss_op
        else:
            loss_op = reward_loss(logits_op, action_var, future_reward_var, sample_weight_var)

        # Add weight decay.
        loss_op += weight_decay * tf.reduce_sum([tf.nn.l2_loss(x) for x in theta])

        opt = tf.train.MomentumOptimizer(lr, 0.9)
        train_op = opt.minimize(loss_op)
        init_op = tf.global_variables_initializer()
        grad_op = tf.gradients(loss_op, theta)

        sess = tf.Session()
        sess.run(init_op)

        def probs_fn(s):
            feed_dict = {state_var: np.reshape(s, (1, state_dim))} # Create batch.
            _, p, v = sess.run([logits_op, probs_op, value_op], feed_dict=feed_dict)
            return p[0], v[0] # Unpack from batch.

        for it in xrange(num_iters):
            # Construct a batch of inputs.
            episodes = []
            for i in xrange(num_episodes):
                ep = run_episode(env, probs_fn, max_time_steps=max_time_steps)
                episodes.append(ep)
                total_num_episodes += 1

            n = sum([len(ep['state']) for ep in episodes])
            # Undo average in mini-batch.
            # Divide by number of episodes.
            weight = [[1.0/len(ep['state']) for t in ep['state']] for ep in episodes]
            total_rewards = [sum(ep['reward']) for ep in episodes]
            future_rewards = [compute_future_rewards(ep['reward'], discount) for ep in episodes]
            values, grads, _ = sess.run([value_op, grad_op, train_op], feed_dict={
                state_var:         np.array(concat([ep['state'] for ep in episodes])),
                action_var:        np.array(concat([ep['action'] for ep in episodes])),
                future_reward_var: np.array(concat(future_rewards)),
                sample_weight_var: np.array(concat(weight)),
            })
            h['reward'].append(np.mean(total_rewards))
            h['num_episodes'].append(total_num_episodes)
            grad_norms = [np.max(np.abs(g)) for g in grads]
            # print 'grads:', ' '.join('%.2e' % x for x in grad_norms)
            h['gradient'].append(grad_norms)
            h['mean_future_reward'].append(np.mean([np.mean(r) for r in future_rewards]))
            h['mean_value'].append(np.mean([np.mean(ep['value']) for ep in episodes]))
            print '%d  reward:%10.3e' % (it, np.mean(total_rewards))

    return h

def compute_future_rewards(r, gamma):
    n = len(r)
    s = 0
    ss = []
    for k in range(n):
        s = r[n-1-k] + gamma*s
        ss.append(s)
    return ss[::-1]

def concat(x):
    return [e for l in x for e in l]

def run_episode(env, policy, render=False, max_time_steps=1000):
    ep = {
        'state':  [],
        'action': [],
        'reward': [],
        'value':  [],
    }
    s_t = env.reset()
    for t in xrange(max_time_steps):
        ep['state'].append(s_t)
        # Evaluate the neural network.
        probs, v_t = policy(s_t)
        ep['value'].append(v_t)
        # Sample the action.
        a_t = np.random.choice(len(probs), p=probs)
        ep['action'].append(a_t)
        s_t, r_t, done, info = env.step(a_t)
        ep['reward'].append(r_t)
        # print 'value:%10.3e probs: %s' % (v_t, ' '.join('%.3f' % p for p in probs))
        if render:
            env.render()
        if done:
            break
    return ep

def reward_loss(logits, actions, future_rewards, weights):
    actions = tf.to_int32(actions)
    # Get the log of the normalized logits for each action.
    # log(exp(logit[action]) / sum(exp(logit)))
    log_p_action = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, actions)
    return tf.reduce_sum(tf.mul(weights, tf.mul(future_rewards, log_p_action)))

def advantage_loss(logits, value, actions, future_rewards, weights):
    actions = tf.to_int32(actions)
    # Get the log of the normalized logits for each action.
    # log(exp(logit[action]) / sum(exp(logit)))
    log_p_action = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, actions)
    return tf.reduce_sum(tf.mul(weights, tf.mul(future_rewards - value, log_p_action)))
