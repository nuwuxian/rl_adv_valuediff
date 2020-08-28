import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

# read the events
def read_events_file(events_filename):
    folders = os.listdir(events_filename)
    events = []
    for folder in folders:
        if os.path.exists(os.path.join(events_filename, folder+'/'+'Log.txt')):
            event = pd.read_csv(os.path.join(events_filename, folder+'/'+'Log.txt'), sep=' ',
                                names=['step', 'aa', 'bb', 'cc', 'win', 'win_1', 'win_plus_tie', 'dd', 'ee', 'ff'],
                                index_col=0)
            events.append(event[0:2200])
    data_form = pd.concat(events)
    data_form = data_form.sort_index()
    data_form = data_form.reset_index()
    data_form_win = data_form.groupby('step')['win']
    data_form_tie = data_form.groupby('step')['win_plus_tie']
    return data_form_win, data_form_tie

# plot the graph
def plot_data(log_dir, out_dir, filename):

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['orangered']

    group_win, group_tie = read_events_file(log_dir)

    min_win, mean_win, max_win = group_win.min(), group_win.mean(), group_win.max()
    min_tie, mean_tie, max_tie = group_tie.min(), group_tie.mean(), group_tie.max()

    ax.fill_between(x=mean_win.index, y1=min_win, y2=max_win, alpha=0.4, color=colors[0])
    mean_win.plot(ax=ax, color=colors[0], linewidth=3)

    # ax.fill_between(x=mean_tie.index, y1=min_tie, y2=max_tie, alpha=0.4, color=colors[0])
    # mean_tie.plot(ax=ax, color=colors[0], linewidth=3)

    ax.set_xticks([0, 40e+4, 80e+4, 120e+4, 160e+4, 200e+4])
    ax.set_yticks([0, 0.5, 1])
    plt.grid(True)
    print_info = []
    print_info.append('%s: min: %.4f, mean: %.4f, max: %.4f.' % ('our', max(min_win), max(mean_win), max(max_win)))
    # print_info.append('%s: min: %.4f, mean: %.4f, max: %.4f.' % ('our', max(min_tie), max(mean_tie), max(max_tie)))
    fig.savefig(out_dir + '/' + filename.split('.')[0]+' '+print_info[0]+'.png')

    # fig, ax = plt.subplots(figsize=(10, 8))
    # group_tie.std().plot(ax=ax, color=colors[0], linewidth=3)
    #
    # ax.set_xticks([0, 40e+4, 80e+4, 120e+4, 160e+4])
    # plt.grid(True)
    # fig.savefig(out_dir + '/' + filename.split('.')[0]+'_std.png')


# main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='/Users/Henryguo/Desktop/rl_newloss/StarCraftII/StarCraft-results/retrain_50/our_attack')
    parser.add_argument('--out_dir', type=str, default='/Users/Henryguo/Desktop/rl_newloss/StarCraftII/StarCraft-results/retrain_50/our_attack')
    parser.add_argument("--filename", type=str, default='results.png')
    args = parser.parse_args()

    out_dir = args.out_dir
    log_dir = args.log_dir
    filename = 'against_adv_wining.png'
    # filename = 'retrain_wining_tie.png'

    plot_data(log_dir=log_dir, out_dir=out_dir, filename=filename)

    # retrain from scratch vs original
    # w = np.array([0.01, 0.00, 0.00, 0.91, 0.00, 0.00])
    # print(np.min(w))
    # print(np.max(w))
    # print(np.mean(w))
    # print(np.std(w))

    # # retrain from scratch vs bot
    # w1 = np.array([0.79, 0.44, 0.05, 0.08, 0.01, 0.04, 0])
    # w2 = np.array([0.5, 0.49, 0.04, 0.01, 0.0, 0.0, 0.0])
    # w3 = np.array([0.55, 0.4, 0.01, 0.03, 0.0, 0.0, 0.0])
    # w4 = np.array([0.67, 0.53, 0.23, 0.08, 0.0, 0.01, 0.28])
    # w5 = np.array([0.56, 0.44, 0.07, 0.0, 0.0, 0.0, 0.0])
    # w6 = np.array([0.52, 0.5, 0.05, 0.01, 0, 0.0, 0.0])
    # # w7 = np.array([0.67, 0.51, 0.07, 0.00, 0, 0.0, 0.08])
    # a = np.vstack((w1, w2, w3, w4, w5, w6))
    # print(np.min(a, axis=0))
    # print(np.max(a, axis=0))
    # print(np.mean(a, axis=0))
    # print(np.std(a, axis=0))

    # # retrain victim vs original
    # w = np.array([0.54, 0.22, 0.00, 0.45, 0.46, 0.7])
    # print(np.min(w))
    # print(np.max(w))
    # print(np.mean(w))
    # print(np.std(w))

    # # retrain victim vs bot
    # w1 = np.array([1, 1, 0.97, 0.97, 0.98, 0.98, 1])
    # w2 = np.array([0.99, 1, 0.97, 0.97, 0.96, 0.93, 0.98])
    # w3 = np.array([0.98, 0.89, 0.85, 0.6, 0.55, 0.61, 0.08])
    # w4 = np.array([1, 1, 0.96, 0.99, 0.98, 0.94, 0.95])
    # w5 = np.array([1, 0.99, 0.96, 0.92, 0.79, 0.79, 1])
    # w6 = np.array([1, 0.98, 0.98, 1, 0.97, 0.98, 1])
    # a = np.vstack((w1, w2, w3, w4, w5, w6))
    # print(np.min(a, axis=0))
    # print(np.max(a, axis=0))
    # print(np.mean(a, axis=0))
    # print(np.std(a, axis=0))
