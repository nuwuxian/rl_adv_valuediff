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
                print(root)
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
def plot_data(log_dir, out_dir, filename, num):

    events = load_tb_data(log_dir, keys=['game_win0'])
    assert len(events) % num == 0

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['r', 'b', 'g', 'purple']

    for i in range(len(events) // num):
        sub_events = events[i*num:(i+1)*num]
        subset = data_frame(sub_events)

        # plot the graph
        group = subset.groupby('step')['game_win0']
        min, median, max = group.min(), group.mean(), group.max()

        ax.fill_between(x=median.index, y1=min, y2=max,\
                        alpha=0.4, color=colors[i])
        median.plot(ax=ax, color=colors[i])

    ax.set_xticks([0, 0.5e+7, 1e+7, 1.5e+7, 2e+7])
    ax.set_yticks([0, 0.5, 0.6, 0.65, 1])
    plt.grid(True)

    fig.savefig(out_dir + '/' + filename)


# main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seed", type=int, default=8)
    parser.add_argument('--log_dir', type=str, default='/home/xkw5132/rl_results/usenix_paper')
    parser.add_argument("--out_dir", type=str, default='/home/xkw5132/rl_results/usenix_paper')
    parser.add_argument("--filename", type=str, default='out.png')
    args = parser.parse_args()

    num = args.num_seed
    out_dir = args.out_dir
    filename = args.filename
    log_dir = args.log_dir

    plot_data(log_dir=log_dir, out_dir=out_dir, \
              filename=filename, num=num)