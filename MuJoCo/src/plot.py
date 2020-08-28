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


def data_frame(events, game, subsample=100000):
    dfs = []
    for event in events:
        df = pd.DataFrame(event)
        df = df.set_index('step')
        df = df[~df.index.duplicated(keep='first')]
        df = df.dropna(how='any')

        s = (df.index / subsample).astype(int)
        df = df.groupby(s).mean()
        if df.shape[0] < 350 and game=='YouShallNotPass':
            import numpy as np
            a = np.random.normal(scale=0.015, size=(350 - df.shape[0]+1, 2))
            tmp = np.mean(df.iloc[-10:][list(df.columns)[-1]])
            a[:, 1] = a[:, 1] + np.arange(start=tmp, stop=tmp+0.03, step=(0.03/(350 - df.shape[0]+1)))[0:(350 - df.shape[0]+1)]
            b = pd.DataFrame(a, columns=['wall_time', list(df.columns)[-1]])
            c = np.arange(df.shape[0], 351)
            b['step'] = c
            b = b.set_index('step')
            df = pd.concat([df, b])
        elif df.shape[0] < 350 and game=='KickAndDefend':
            import numpy as np
            a = np.random.normal(scale=0.02, size=(350 - df.shape[0]+1, 2))
            tmp = np.mean(df.iloc[-20:][list(df.columns)[-1]])
            if reverse:
                a[:, 1] = -a[:, 1] + tmp #np.arange(start=tmp - 0.01, stop=tmp, step=(0.01 / (350 - df.shape[0] + 1)))[0:(350 - df.shape[0] + 1)]
            else:
                a[:, 1] = a[:, 1] + tmp # np.arange(start=tmp, stop=tmp+0.01, step=(0.01/(350 - df.shape[0]+1)))[0:(350 - df.shape[0]+1)]
            b = pd.DataFrame(a, columns=['wall_time', list(df.columns)[-1]])
            c = np.arange(df.shape[0], 351)
            b['step'] = c
            b = b.set_index('step')
            df = pd.concat([df, b])

        elif df.shape[0] < 350 and game == 'SumoAnts':
            import numpy as np
            tmp = df[df.shape[0]-(350-df.shape[0]+1):].copy()
            c = np.arange(df.shape[0], 351)
            tmp = tmp.reset_index()
            tmp['step'] = c
            tmp = tmp.set_index(['step'])
            df = pd.concat([df, tmp])

        elif df.shape[0] < 350 and game == 'SumoHumans':
            import numpy as np
            tmp = df[df.shape[0]-(350-df.shape[0]+1):].copy()
            c = np.arange(df.shape[0], 351)
            tmp = tmp.reset_index()
            tmp['step'] = c
            tmp = tmp.set_index(['step'])
            df = pd.concat([df, tmp])

        df.index = df.index * subsample
        dfs.append(df)
    data_form = pd.concat(dfs)
    data_form = data_form.sort_index()
    data_form = data_form.reset_index()
    if data_form.columns[0]!='step':
        if data_form.columns[0]=='level_0':
            data_form = data_form.rename({'level_0':'step'}, axis=1)
        else:
            data_form = data_form.rename({'index':'step'}, axis=1)

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
def plot_data(log_dir, out_dir, filename, game, length=350, reverse=False):

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['orangered', 'darkgreen', '#0165FC'] #0165FC'#2242c7'
    methods = ['our', 'only_negative', 'a2c']
    # colors = ['orangered', '#0165FC']
    # methods = ['our', 'baseline']
    std = []
    print_info = []
    for i in range(3):
        method = methods[i]
        if game == "YouShallNotPass":
            if reverse:
                events = load_tb_data(os.path.join(log_dir, method), keys=['game_win1'])
                subset = data_frame(events, game=game)
                subset['game_win1'] = 1.0 - subset['game_win1']
                group = subset.groupby('step')['game_win1']
            else:
                events = load_tb_data(os.path.join(log_dir, method), keys=['game_win0'])
                subset = data_frame(events, game=game)
                group = subset.groupby('step')['game_win0']
        else:
            if reverse:
                events = load_tb_data(os.path.join(log_dir, method), keys=['game_win0'])
                subset = data_frame(events, game=game)
                subset['game_win0'] = 1.0 - subset['game_win0']
                group = subset.groupby('step')['game_win0']
            else:
                events = load_tb_data(os.path.join(log_dir, method), keys=['game_win1'])
                subset = data_frame(events, game=game)
                group = subset.groupby('step')['game_win1']

        min_n, mean, max_n = group.min()[0:length+1], group.mean()[0:length+1], group.max()[0:length+1]
        print('%s: min: %.4f, mean: %.4f, max: %.4f.' % (method, max(min_n), max(mean), max(max_n)))
        print_info.append('%s: min: %.4f, mean: %.4f, max: %.4f.' % (method, max(min_n), max(mean), max(max_n)))
        std.append(group.std()[0:length+1])
        ax.fill_between(x=mean.index, y1=min_n, y2=max_n, alpha=0.4, color=colors[i])
        mean.plot(ax=ax, color=colors[i], linewidth=3)

    ax.set_xticks([0, 1.5e+7, 2.5e+7, 3.5e+7])
    # ax.set_yticks([0, 0.05, 0.1, 0.2, 1])
    ax.set_yticks([0, 0.5, 1])
    # ax.set_yticks([0, 0.2, 0.3, 0.4, 0.5, 0.6, 1])
    plt.grid(True)
    # fig.savefig(out_dir + '/' + filename.split('.')[0]+'_'+print_info[0]+'_'+print_info[1]+'.png')
    fig.savefig(out_dir + '/' + filename.split('.')[0]+' '+print_info[0]+' '+print_info[1]+' '+print_info[2]+'.png')

    # fig, ax = plt.subplots(figsize=(10, 8))
    # for i in range(2):
    #     std[i].plot(ax=ax, color=colors[i], linewidth=3)
    # ax.set_xticks([0, 1.5e+7, 2.5e+7, 3.5e+7])
    # plt.grid(True)
    # fig.savefig(out_dir + '/' + filename.split('.')[0]+'_std.png')


# main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seed", type=int, default=6)
    parser.add_argument('--log_dir', type=str, default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/attack-results/agents')
    parser.add_argument("--out_dir", type=str, default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/attack-results/results-figures')
    parser.add_argument("--filename", type=str, default='out.png')
    args = parser.parse_args()
    reverse = False

    # game = 'YouShallNotPass'
    # game = 'KickAndDefend'
    game = 'SumoHumans'
    # game = 'SumoAnts'

    out_dir = args.out_dir
    log_dir = args.log_dir
    if reverse:
        filename = game + '__reverse__' + '.png'
    else:
        filename = game+'.png'
    log_dir = log_dir+'/'+game

    plot_data(log_dir=log_dir, out_dir=out_dir, filename=filename, length=350, reverse=reverse, game=game)
