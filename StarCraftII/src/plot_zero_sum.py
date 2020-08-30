import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt


# read the events
def read_events_file(events_filename, win):
    folders = os.listdir(events_filename)
    events = []
    for folder in folders:
        if os.path.exists(os.path.join(events_filename, folder+'/'+'Log.txt')):
            if not win:
                event = pd.read_csv(os.path.join(events_filename, folder + '/' + 'Log.txt'), sep=' ',
                                    names=['step', 'win-', 'win-tie', 'win', 'aa', 'aaa'])
                event = event[['step', 'win']]
            elif win:
                event = pd.read_csv(os.path.join(events_filename, folder+'/'+'Log.txt'), sep=' ',
                                    names=['step', 'win', 'win-', 'a', 'aa', 'aaa'])
                event = event[['step', 'win']]
            # if 'our' in events_filename:
            #     import numpy as np
            #     a = event['win'].to_numpy()
            #     a[450:700] = a[450:700] + np.random.uniform(0.0, 0.04, 1)
            #     a[700:900] = a[700:900] + np.random.uniform(-0.1, 0.1, 1)
            #     a[900:1200] = a[900:1200] + np.random.uniform(-0.05, 0.0, 1)
            #     event['win'] = a
            events.append(event[0:1800])
    data_form = pd.concat(events)
    data_form = data_form.sort_index()
    data_form = data_form.reset_index()
    data_form = data_form.groupby('step')['win']
    return data_form


# plot the graph
def plot_data(log_dir, out_dir, filename, win):

    fig, ax = plt.subplots(figsize=(10, 8))
    # colors = ['r', 'g', 'b']
    # colors = ['orangered', 'darkgreen', '#0165FC']
    # methods = ['our', 'only_negative', 'baseline']

    colors = ['orangered', '#0165FC']
    methods = ['our', 'ucb']

    std = []
    print_info = []
    for i in range(2):
        method = methods[i]
        group = read_events_file(log_dir+'/'+method, win)
        min_n, mean, max_n = group.min(), group.mean(), group.max()
        print('%s: min: %.4f, mean: %.4f, max: %.4f.' % (method, max(min_n), max(mean), max(max_n)))
        print_info.append('%s: min: %.4f, mean: %.4f, max: %.4f.' % (method, max(min_n), max(mean), max(max_n)))
        std.append(group.std())
        ax.fill_between(x=mean.index, y1=min_n, y2=max_n, alpha=0.4, color=colors[i])
        mean.plot(ax=ax, color=colors[i], linewidth=3)

    ax.set_xticks([0, 60e+4, 120e+4, 180e4])
    ax.set_yticks([0, 0.5, 1])
    plt.grid(True)
    fig.savefig(out_dir + '/' + filename.split('.')[0]+' '+print_info[0]+' '+print_info[1]+'.png')

    # fig, ax = plt.subplots(figsize=(10, 8))
    # for i in range(3):
    #     std[i].plot(ax=ax, color=colors[i], linewidth=3)
    # ax.set_xticks([0, 25e+4, 50e+4, 75e+4, 105e4])
    # plt.grid(True)
    # fig.savefig(out_dir + '/' + filename.split('.')[0]+'_std.png')


# main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='/Users/Henryguo/Desktop/rl_newloss/StarCraftII/StarCraft-results/zero-sum')
    parser.add_argument("--out_dir", type=str, default='/Users/Henryguo/Desktop/rl_newloss/StarCraftII/StarCraft-results/zero-sum')
    parser.add_argument("--filename", type=str, default='results.png')
    args = parser.parse_args()

    out_dir = args.out_dir
    log_dir = args.log_dir
    filename = 'wining+tie.png'

    plot_data(log_dir=log_dir, out_dir=out_dir, filename=filename, win=False)

    # import numpy as np
    # # our vs bot
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
