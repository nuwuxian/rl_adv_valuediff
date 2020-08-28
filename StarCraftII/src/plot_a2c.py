import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt


# read the events
def read_events_file(events_filename, win):
    folders = os.listdir(events_filename)
    events = []
    for folder in folders:
        print(folder)
        if os.path.exists(os.path.join(events_filename, folder+'/'+'Log.txt')):
            if not win and 'a2c' in events_filename:
                event = pd.read_csv(os.path.join(events_filename, folder + '/' + 'Log.txt'), sep=' ',
                                    names=['step', 'win-', 'win-tie', 'win', 'aa', 'aaa'])
                event = event[['step', 'win']]
                event = event.set_index('step')
            elif win and 'a2c' in events_filename:
                event = pd.read_csv(os.path.join(events_filename, folder + '/' + 'Log.txt'), sep=' ',
                                    names=['step', 'win', 'win-tie', 'win+tie', 'aa', 'aaa'])
                event = event[['step', 'win']]
                event = event.set_index('step')
            else:
                event = pd.read_csv(os.path.join(events_filename, folder+'/'+'Log.txt'), sep=' ', names=['step', 'win'],
                                index_col=0)
            events.append(event[0:1049])
    data_form = pd.concat(events)
    data_form = data_form.sort_index()
    data_form = data_form.reset_index()
    data_form = data_form.groupby('step')['win']
    return data_form


# plot the graph
def plot_data(log_dir, out_dir, filename, win=True):

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['orangered', 'darkgreen', '#0165FC']
    methods = ['our', 'only_negative', 'a2c']

    std = []
    print_info = []
    for i in range(3):
        method = methods[i]
        group = read_events_file(log_dir+'/'+method, win)
        min_n, mean, max_n = group.min(), group.mean(), group.max()
        print('%s: min: %.4f, mean: %.4f, max: %.4f.' % (method, max(min_n), max(mean), max(max_n)))
        print_info.append('%s: min: %.4f, mean: %.4f, max: %.4f.' % (method, max(min_n), max(mean), max(max_n)))
        std.append(group.std())
        ax.fill_between(x=mean.index, y1=min_n, y2=max_n, alpha=0.4, color=colors[i])
        mean.plot(ax=ax, color=colors[i], linewidth=3)

    ax.set_xticks([0, 25e+4, 50e+4, 75e+4, 105e4])
    ax.set_yticks([0, 0.5, 1])
    plt.grid(True)
    fig.savefig(out_dir + '/' + filename.split('.')[0]+' '+print_info[0]+' '+print_info[1]+' '+print_info[2]+'.png')

    # fig, ax = plt.subplots(figsize=(10, 8))
    # for i in range(3):
    #     std[i].plot(ax=ax, color=colors[i], linewidth=3)
    # ax.set_xticks([0, 25e+4, 50e+4, 75e+4, 105e4])
    # plt.grid(True)
    # fig.savefig(out_dir + '/' + filename.split('.')[0]+'_std.png')



# main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='/Users/Henryguo/Desktop/rl_newloss/StarCraftII/StarCraft-results/agents')
    parser.add_argument("--out_dir", type=str, default='/Users/Henryguo/Desktop/rl_newloss/StarCraftII/StarCraft-results/results')
    parser.add_argument("--filename", type=str, default='results.png')
    args = parser.parse_args()

    out_dir = args.out_dir
    log_dir = args.log_dir
    filename = 'wining.png'

    plot_data(log_dir=log_dir, out_dir=out_dir, filename=filename, win=False)
