import os, os.path
import csv
import jinja2
import subprocess
import gym

import pg
import cartpole

def main():
    num_trials    = 8
    num_iters     = 100
    out_dir       = 'out'
    use_advantage = True
    num_episodes  = 30
    learning_rate = 1e-8
    discounts     = 0.9

    env = gym.make('CartPole-v0')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    trial_names = []
    for trial in range(num_trials):
        trial_name = 'trial-%d' % trial
        create_policy = lambda x: cartpole.create_policy(x, input_dim=4, action_dim=2, use_value=use_advantage)
        hist = pg.train(env, create_policy,
            state_dim=4, action_dim=2,
            num_iters=num_iters,
            num_episodes=num_episodes,
            lr=learning_rate,
            discount=discounts,
            use_advantage=use_advantage,
            max_time_steps=1000)
        write_data(os.path.join(out_dir, trial_name+'.tsv'), hist)
        trial_names.append(trial_name)
    # Plot all trials together.
    plot(out_dir, 'pg', trial_names)
    plot_each(out_dir, trial_names, range(6))

def write_data(fname, hist):
    with open(fname, 'wb') as f:
        w = csv.writer(f, delimiter='\t')
        for t in range(len(hist['reward'])):
            row = [hist['num_episodes'][t], hist['reward'][t]] + hist['gradient'][t]
            w.writerow(row)

def plot(out_dir, name, trial_names):
    template = jinja2.Template('''
set terminal png
set output '{{ name }}.png'
plot {% for trial in trials %}'{{ trial }}.tsv' using 1:2 title '{{ trial }}' with lines{% if not loop.last %}, \\
{% endif %}{% endfor %}
''')
    s = template.render(name=name, trials=trial_names)
    script_file = name+'.gnuplot'
    with open(os.path.join(out_dir, script_file), 'wb') as f:
        f.write(s)
    p = subprocess.Popen(['gnuplot', script_file], cwd=out_dir)
    p.wait()

def plot_each(out_dir, trial_names, param_inds):
    template = jinja2.Template('''
set terminal png
set output '{{ name }}.png'
set y2tics
set logscale y2
set format y2 '%.1g'
plot '{{ name }}.tsv' using 1:2 title 'reward' with lines axes x1y1, \\
{% for param in params %}'' using 1:{{ 3+param }} title 'grad {{ param }}' with lines lt 0 lc {{ 2+loop.index }} axes x1y2{% if not loop.last %}, \\
{% endif %}{% endfor %}
''')
    for trial in trial_names:
        s = template.render(name=trial, params=param_inds)
        script_file = trial+'.gnuplot'
        with open(os.path.join(out_dir, script_file), 'wb') as f:
            f.write(s)
        p = subprocess.Popen(['gnuplot', script_file], cwd=out_dir)
        p.wait()

if __name__ == '__main__':
    main()
