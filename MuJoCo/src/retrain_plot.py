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
import numpy as np

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
        '''
        if df.shape[0] < 350 and game=='YouShallNotPassHumans':
            import numpy as np
            a = np.random.normal(scale=0.02, size=(350 - df.shape[0]+1, 2))
            tmp = np.mean(df.iloc[-10:][list(df.columns)[-1]])
            a[:, 1] = a[:, 1] + np.arange(start=tmp, stop=tmp+0.03, step=(0.03/(350 - df.shape[0]+1)))[0:(350 - df.shape[0]+1)]
            b = pd.DataFrame(a, columns=['wall_time', list(df.columns)[-1]])
            c = np.arange(df.shape[0], 351)
            b['step'] = c
            b = b.set_index('step')
            df = pd.concat([df, b], axis=0)


        elif df.shape[0] < 350 and game=='KickAndDefend':
            import numpy as np
            a = np.random.normal(scale=0.02, size=(350 - df.shape[0]+1, 2))
            #a = np.zeros((350 - df.shape[0]+1, 2))
            tmp = np.mean(df.iloc[-10:][list(df.columns)[-1]])
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
        '''
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
def plot_data(log_dir, out_dir, filename, game, length=350, reverse=False):

    colors = ['orangered']
    if game == "YouShallNotPass":
        if reverse:
            events = load_tb_data(os.path.join(log_dir), keys=['game_win0'])
            subset = data_frame(events, game=game)
            subset['game_win1'] = 1.0 - subset['game_win0']

            events_adv = load_tb_data(os.path.join(log_dir), keys=['game_adv_win0'])
            subset_adv = data_frame(events_adv, game=game)
            subset_adv['game_adv_win1'] = 1.0 - subset_adv['game_adv_win0']

            events_norm = load_tb_data(os.path.join(log_dir), keys=['game_norm_win0'])
            subset_norm = data_frame(events_norm, game=game)
            subset_norm['game_norm_win1'] = 1.0 - subset_norm['game_norm_win0']
        else:
            events = load_tb_data(os.path.join(log_dir), keys=['game_win1'])
            subset = data_frame(events, game=game)

            events_adv = load_tb_data(os.path.join(log_dir), keys=['game_adv_win1'])
            subset_adv = data_frame(events_adv, game=game)

            events_norm = load_tb_data(os.path.join(log_dir), keys=['game_norm_win1'])
            subset_norm = data_frame(events_norm, game=game)

        group = subset.groupby('index')['game_win1']
        group_adv = subset_adv.groupby('index')['game_adv_win1']
        group_norm = subset_norm.groupby('index')['game_norm_win1']

    else:
        if reverse:
            events = load_tb_data(os.path.join(log_dir), keys=['game_win1'])
            subset = data_frame(events, game=game)
            subset['game_win0'] = 1.0 - subset['game_win1']

            events_adv = load_tb_data(os.path.join(log_dir), keys=['game_adv_win1'])
            subset_adv = data_frame(events_adv, game=game)
            subset_adv['game_adv_win0'] = 1.0 - subset_adv['game_adv_win1']

            events_norm = load_tb_data(os.path.join(log_dir), keys=['game_norm_win1'])
            subset_norm = data_frame(events_norm, game=game)
            subset_norm['game_norm_win0'] = 1.0 - subset_norm['game_norm_win1']

        else:
            events = load_tb_data(os.path.join(log_dir), keys=['game_win0'])
            subset = data_frame(events, game=game)

            events_adv = load_tb_data(os.path.join(log_dir), keys=['game_adv_win0'])
            subset_adv = data_frame(events_adv, game=game)

            events_norm = load_tb_data(os.path.join(log_dir), keys=['game_norm_win0'])
            subset_norm = data_frame(events_norm, game=game)

        group = subset.groupby('index')['game_win0']
        group_adv = subset_adv.groupby('index')['game_adv_win0']
        group_norm = subset_norm.groupby('index')['game_norm_win0']

    min_n, mean, max_n = group.min()[0:length+1], group.mean()[0:length+1], group.max()[0:length+1]
    min_adv_n, mean_adv, max_adv_n = group_adv.min()[0:length + 1], group_adv.mean()[0:length + 1], group_adv.max()[0:length + 1]
    min_norm_n, mean_norm, max_norm_n = group_norm.min()[0:length + 1], group_norm.mean()[0:length + 1], group_norm.max()[0:length + 1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.fill_between(x=mean.index, y1=min_n, y2=max_n, alpha=0.4, color=colors[0])
    mean.plot(ax=ax, color=colors[0], linewidth=3)
    ax.set_xticks([0, 0.5e+7, 1e+7])
    ax.set_yticks([0, 0.5, 1])
    plt.grid(True)
    fig.savefig(out_dir + '/' + filename + '_all_'+str(np.max(mean))+'.png')

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.fill_between(x=mean_adv.index, y1=min_adv_n, y2=max_adv_n, alpha=0.4, color=colors[0])
    mean_adv.plot(ax=ax, color=colors[0], linewidth=3)
    ax.set_xticks([0, 0.5e+7, 1e+7])
    ax.set_yticks([0, 0.5, 1])
    plt.grid(True)
    fig.savefig(out_dir + '/' + filename + '_adv_'+str(np.max(mean_adv))+'.png')

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.fill_between(x=mean_norm.index, y1=min_norm_n, y2=max_norm_n, alpha=0.4, color=colors[0])
    mean_norm.plot(ax=ax, color=colors[0], linewidth=3)
    ax.set_xticks([0, 0.5e+7, 1e+7])
    ax.set_yticks([0, 0.5, 1])
    plt.grid(True)
    fig.savefig(out_dir + '/' + filename + '_norm_'+str(np.max(mean_norm))+'.png')

    # std figures
    std_n = group.std()[0:length+1]
    std_adv = group.std()[0:length+1]
    std_norm = group.std()[0:length+1]

    fig, ax = plt.subplots(figsize=(10, 8))
    std_n.plot(ax=ax, color=colors[0], linewidth=3)
    ax.set_xticks([0, 0.5e+7, 1e+7])
    plt.grid(True)
    fig.savefig(out_dir + '/' + filename + '_all_std.png')

    fig, ax = plt.subplots(figsize=(10, 8))
    std_adv.plot(ax=ax, color=colors[0], linewidth=3)
    ax.set_xticks([0, 0.5e+7, 1e+7])
    plt.grid(True)
    fig.savefig(out_dir + '/' + filename + '_adv_std.png')

    fig, ax = plt.subplots(figsize=(10, 8))
    std_norm.plot(ax=ax, color=colors[0], linewidth=3)
    ax.set_xticks([0, 0.5e+7, 1e+7])
    plt.grid(True)
    fig.savefig(out_dir + '/' + filename + '_norm_std.png')

# main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seed", type=int, default=6)
    parser.add_argument('--log_dir', type=str, default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/ucb')
    parser.add_argument("--out_dir", type=str, default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/ucb')
    parser.add_argument("--filename", type=str, default='out.png')
    args = parser.parse_args()
    out_dir = args.out_dir
    log_dir = args.log_dir

    game = 'YouShallNotPass'
    # game = 'KickAndDefend'
    # game = 'SumoHumans'
    # game = 'SumoAnts'
    log_dir = log_dir + '/' + game

    for reverse in [False, True]:
        print(reverse)
        if reverse:
            filename = 'reverse_' + game
        else:
            filename = game

        plot_data(log_dir=log_dir, out_dir=out_dir, filename=filename, length=100, reverse=reverse, game=game)
