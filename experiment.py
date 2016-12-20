import os, os.path
import csv
import jinja2
import subprocess
import gym

import policygrad
import cartpole

def main():
    num_trials    = 8
    num_iters     = 100
    out_dir       = 'out'
    use_advantage = True

    num_episodes  = [10, 30]
    learning_rate = [1e-9, 1e-7, 1e-5]
    discounts     = [0.9, 0.99, 1.0]

    env = gym.make('CartPole-v0')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i in range(len(num_episodes)):
        for k in range(len(discounts)):
            for j in range(len(learning_rate)):
                name = 'episodes-%d-lr-%g-discount-%g' % (
                    num_episodes[i], learning_rate[j], discounts[k])
                trial_names = []
                for trial in range(num_trials):
                    trial_name = '%s-trial-%d' % (name, trial)
                    create_policy = lambda x: cartpole.create_policy(x, use_value=use_advantage)
                    hist = policygrad.train(env, create_policy,
                        state_dim=4, action_dim=2,
                        num_iters=num_iters,
                        num_episodes=num_episodes[i],
                        lr=learning_rate[j],
                        discount=discounts[k],
                        use_advantage=use_advantage)
                    write_data(os.path.join(out_dir, trial_name+'.tsv'), hist)
                    trial_names.append(trial_name)
                # Plot all trials together.
                plot(out_dir, name, trial_names)

def write_data(fname, hist):
    with open(fname, 'wb') as f:
        w = csv.writer(f, delimiter='\t')
        for t in range(len(hist['reward'])):
            w.writerow([hist['num_episodes'][t], hist['reward'][t]])

def plot(out_dir, name, trial_names):
    template = jinja2.Template('''
set terminal png
set output '{{ name }}.png'
plot {% for trial in trials %}'{{ trial }}.tsv' using 1:2 title '{{trial}}' with lines{% if not loop.last %}, \\
{% endif %}{% endfor %}
''')
    s = template.render(name=name, trials=trial_names)
    script_file = name+'.gnuplot'
    with open(os.path.join(out_dir, script_file), 'wb') as f:
        f.write(s)
    p = subprocess.Popen(['gnuplot', script_file], cwd=out_dir)
    p.wait()

if __name__ == '__main__':
    main()
