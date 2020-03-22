import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt

# read the events
def read_events_file(events_filename):
    folders = os.listdir(events_filename)
    events = []
    for folder in folders:
        if os.path.exists(os.path.join(events_filename, folder+'/'+'Log.txt')):
            event = pd.read_csv(os.path.join(events_filename, folder+'/'+'Log.txt'), sep=' ',
                                names=['step', 'win', 'win_1', 'win_plus_tie'],
                                index_col=0)
            events.append(event[0:1600])
    data_form = pd.concat(events)
    data_form = data_form.sort_index()
    data_form = data_form.reset_index()
    data_form_win = data_form.groupby('step')['win']
    data_form_tie = data_form.groupby('step')['win_plus_tie']
    return data_form_win, data_form_tie

# plot the graph
def plot_data(log_dir, out_dir, filename):

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['r', 'b']

    group_win, group_tie = read_events_file(log_dir+'/'+'retrain-victim')

    min_win, mean_win, max_win = group_win.min(), group_win.mean(), group_win.max()
    min_tie, mean_tie, max_tie = group_tie.min(), group_tie.mean(), group_tie.max()

    ax.fill_between(x=mean_win.index, y1=min_win, y2=max_win, alpha=0.4, color=colors[0])
    mean_win.plot(ax=ax, color=colors[0], linewidth=3)

    ax.fill_between(x=mean_tie.index, y1=min_tie, y2=max_tie, alpha=0.4, color=colors[1])
    mean_tie.plot(ax=ax, color=colors[1], linewidth=3)

    ax.set_xticks([0, 40e+4, 80e+4, 120e+4, 160e+4])
    ax.set_yticks([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 0.95, 1])
    plt.grid(True)
    fig.savefig(out_dir + '/' + filename)


    fig, ax = plt.subplots(figsize=(10, 8))
    group_win.std().plot(ax=ax, color=colors[0], linewidth=3)
    group_tie.std().plot(ax=ax, color=colors[1], linewidth=3)

    ax.set_xticks([0, 40e+4, 80e+4, 120e+4, 160e+4])
    plt.grid(True)
    fig.savefig(out_dir + '/' + filename.split('.')[0]+'_std.png')


# main function
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--log_dir', type=str, default='/Users/Henryguo/Desktop/StarCraft-results/agents')
    # parser.add_argument("--out_dir", type=str, default='/Users/Henryguo/Desktop/StarCraft-results/results')
    # parser.add_argument("--filename", type=str, default='results.png')
    # args = parser.parse_args()
    #
    # out_dir = args.out_dir
    # log_dir = args.log_dir
    # filename = 'retrain_wining.png'
    #
    # plot_data(log_dir=log_dir, out_dir=out_dir, filename=filename)
    #

    import numpy as np
    # # retrain vs original
    w = np.array([0.54, 0.22, 0.00, 0.45, 0.46, 0.7])
    print(np.min(w))
    print(np.max(w))
    print(np.mean(w))
    print(np.std(w))

    # w1 = np.array([1, 0.84, 0.51, 0.21, 0.05, 0.03, 0.00])
    # w2 = np.array([1, 0.91, 0.56, 0.16, 0.03, 0.01, 0.00])
    # w3 = np.array([0.5, 0.5, 0.48, 0.14, 0.04, 0.00, 0.00])
    # w4 = np.array([1, 0.92, 0.38, 0.20, 0.02, 0.04, 0.00])
    # w5 = np.array([0.58, 0.5, 0.46, 0.22, 0.03, 0.02, 0.00])
    # w6 = np.array([0.99, 0.75, 0.55, 0.12, 0.03, 0.03, 0.00])
    # a = np.vstack((w1, w2, w3, w4, w5, w6))
    # print(np.min(a, axis=0))
    # print(np.max(a, axis=0))
    # print(np.mean(a, axis=0))
    # print(np.std(a, axis=0))
    #
    # # ucb vs bot
    # w1 = np.array([0.57, 0.50, 0.23, 0.20, 0.00, 0.02, 0.00])
    # w2 = np.array([0.66, 0.48, 0.15, 0.14, 0.03, 0.02, 0.00])
    # w3 = np.array([0.5, 0.34, 0.11, 0.20, 0.01, 0.02, 0.00])
    # w4 = np.array([0.53, 0.39, 0.05, 0.15, 0.05, 0.03, 0.00])
    # w5 = np.array([0.51, 0.28, 0.04, 0.22, 0.04, 0.01, 0.00])
    # w6 = np.array([0.73, 0.48, 0.18, 0.14, 0.05, 0.06, 0.00])
    #
    # a = np.vstack((w1, w2, w3, w4, w5, w6))
    # print(np.min(a, axis=0))
    # print(np.max(a, axis=0))
    # print(np.mean(a, axis=0))
    # print(np.std(a, axis=0))
