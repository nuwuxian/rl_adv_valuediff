import os
import fnmatch
import tensorflow as tf
import logging
import functools
import traceback
import multiprocessing
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import pdb

logger = logging.getLogger('modelfree.visualize.tb')


# find the file
def find_tfevents(log_dir):
    result = []
    for root, dirs, files in os.walk(log_dir, followlinks=True):
        if root.endswith('rl/tb'):
            for name in files:
                # print(root)
                if fnmatch.fnmatch(name, 'events.out.tfevents.*'):
                    result.append(os.path.join(root, name))
    return result


# read the events
def read_events_file(events_filename, keys=None):
    events = []
    try:
        for event in tf.train.summary_iterator(events_filename):
            row = {'wall_time': event.wall_time, 'step': event.step}
            for value in event.summary.value:
                if keys is not None and value.tag not in keys:
                    continue
                row[value.tag] = value.simple_value
            events.append(row)
    except Exception:
        logger.error(f"While reading '{events_filename}': {traceback.print_exc()}")
    return events


def data_frame(events, subsample=100000):
    dfs = []
    for event in events:
        df = pd.DataFrame(event)
        df = df.set_index('step')
        df = df[~df.index.duplicated(keep='first')]
        df = df.dropna(how='any')

        s = (df.index / subsample).astype(int)
        df = df.groupby(s).mean()

        df.index = df.index * subsample
        dfs.append(df)
    data_form = pd.concat(dfs)
    data_form = data_form.sort_index()
    data_form = data_form.reset_index()
    return data_form


# read the tb data
# set the data form
# suppose we have the
def load_tb_data(log_dir, keys=None):
    event_paths = find_tfevents(log_dir)
    pool = multiprocessing.Pool()
    events_by_path = pool.map(functools.partial(read_events_file, keys=keys), event_paths)
    return events_by_path


# plot the graph
def plot_data(log_dir, out_dir, filename, length):

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['r', 'g', 'b']
    # colors = ['r', 'b', 'g', 'purple']
    methods = ['our', 'only_negative', 'baseline']
    std = []
    for i in range(3):
        method = methods[i]
        if game=="YouShallNotPassHumans":
            events = load_tb_data(os.path.join(log_dir, method), keys=['game_win0'])
            subset = data_frame(events)
            group = subset.groupby('index')['game_win0']
        else:
            events = load_tb_data(os.path.join(log_dir, method), keys=['game_win1'])
            subset = data_frame(events)
            group = subset.groupby('index')['game_win1']

        min_n, mean, max_n = group.min()[0:length+1], group.mean()[0:length+1], group.max()[0:length+1]
        print('%s: min: %.4f, mean: %.4f, max: %.4f.' % (method, max(min_n), max(mean), max(max_n)))
        std.append(group.std()[0:length+1])
        ax.fill_between(x=mean.index, y1=min_n, y2=max_n, alpha=0.4, color=colors[i])
        mean.plot(ax=ax, color=colors[i], linewidth=3)

    ax.set_xticks([0, 0.5e+7, 1e+7, 1.5e+7, 2e+7, 2.5e+7, 3e+7, 3.5e+7])
    ax.set_yticks([0, 0.2, 0.3, 0.4, 0.5, 0.6, 1])
    plt.grid(True)
    fig.savefig(out_dir + '/' + filename)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(3):
        std[i].plot(ax=ax, color=colors[i], linewidth=3)
    ax.set_xticks([0, 0.5e+7, 1e+7, 1.5e+7, 2e+7, 2.5e+7, 3e+7, 3.5e+7])
    plt.grid(True)
    fig.savefig(out_dir + '/' + filename.split('.')[0]+'_std.png')


# main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seed", type=int, default=6)
    parser.add_argument('--log_dir', type=str, default='/Users/Henryguo/Desktop/agent-zoo')
    parser.add_argument("--out_dir", type=str, default='/Users/Henryguo/Desktop/results')
    parser.add_argument("--filename", type=str, default='out.png')
    args = parser.parse_args()

    # game = 'YouShallNotPassHumans'
    # game = 'KickAndDefend'
    # game = 'SumoHumans'
    game = 'SumoAnts'

    out_dir = args.out_dir
    log_dir = args.log_dir
    filename = game+'.png'
    log_dir = log_dir+'/'+game

    plot_data(log_dir=log_dir, out_dir=out_dir, filename=filename, length=350)
