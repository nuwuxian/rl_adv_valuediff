import collections
import argparse
import os
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

TRAIN_ID = 'ucb'

ENV_NAMES = ["KickAndDefend", "SumoHumans", "SumoAnts", "YouShallNotPassHumans"]

PRETTY_ENVS = collections.OrderedDict(
    [
        ("KickAndDefend", "Kick and\nDefend"),
        ("YouShallNotPassHumans", "You Shall\nNot Pass"),
        ("SumoHumans", "Sumo\nHumans"),
        ("SumoAnts", "Sumo\nAnts"),
    ]
)

PRETTY_OPPONENTS = collections.OrderedDict(
    [
        ("ucb", "UcbT"),
        ("ucb_test", "UcbV"),
        ("our", "Our"),
        ("norm", "Zoo"),
        # ("norm", "ZooT"),
        # ("norm_test", "ZooV"),
        # ("our", "Our"),
        # ("ucb", "Ucb"),
    ]
)

CYCLE_ORDER = ["Our", "Zoo", "UcbT", "UcbV"]
BAR_ORDER = ["UcbT", "UcbV", "Our", "Zoo"]
#
# CYCLE_ORDER = ["Our", "Ucb", "ZooT", "ZooV"]
# BAR_ORDER = ["ZooT", "ZooV", "Our", "Ucb"]

STYLES = {
    "paper": {
        "figure.figsize": (11, 15),
        "font.family": "sans-serif",
        "font.serif": "Times New Roman",
        "font.weight": "bold",
        "font.size": 6,
        "legend.fontsize": 6,
        "axes.unicode_minus": False,  # workaround bug with Unicode minus signs not appearing
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
    },
    "density_twocol": {"figure.figsize": (10.8, 8.25), "legend.fontsize": 8},
}

def outside_legend(
    legend_entries,
    legend_ncol,
    fig,
    ax_left,
    ax_right,
    legend_padding=0.25,
    legend_height=0.3,
    **kwargs,
):
    width, height = fig.get_size_inches()

    pos_left = ax_left.get_position(original=True)
    pos_right = ax_right.get_position(original=True)
    legend_left = pos_left.x0
    legend_right = pos_right.x0 + pos_right.width
    legend_width = legend_right - legend_left
    legend_bottom = pos_left.y0 + pos_left.height + legend_padding / height
    legend_height = legend_height / height
    bbox = (legend_left, legend_bottom, legend_width, legend_height)
    fig.legend(
        *legend_entries,
        loc="lower center",
        ncol=legend_ncol,
        bbox_to_anchor=bbox,
        mode="expand",
        borderaxespad=0,
        frameon=True,
        **kwargs,
    )

def density_fitter(activation_paths, output_dir, train_opponent, n_components, type):

    activations = []
    metadata = []
    # norm, ucb, our
    for opponent_id, path in activation_paths.items():
        act = np.load(path, allow_pickle=True)
        activations.append(act)

        opponent_id = [opponent_id] * act.shape[0]
        meta = pd.DataFrame(
                {'opponent_id': opponent_id}
            )
        metadata.append(meta)

    activations = np.concatenate(activations)
    metadata = pd.concat(metadata)

    # prepare the training data
    # Flatten activations (but preserve timestep)
    activations = activations.reshape(activations.shape[0], -1)

    max_timesteps = len(metadata)

    activations = activations[0:max_timesteps]
    metadata = metadata[0:max_timesteps].copy()

    # Split into train, validation and test
    opponent_mask = metadata["opponent_id"] == train_opponent
    percentage_mask = np.random.choice(
        [True, False], size=len(metadata), p=[0.5, 0.5]
    )

    metadata["is_train"] = opponent_mask & percentage_mask
    train_data = activations[metadata["is_train"]]

    train_opponent_validation_mask = opponent_mask & ~percentage_mask
    train_opponent_validation_data = activations[train_opponent_validation_mask]

    # Fit model and evaluate
    model_kwargs = {"n_components": n_components, "covariance_type": type}

    model_obj = GaussianMixture(**model_kwargs)
    model_obj.fit(train_data)
    metadata["log_proba"] = model_obj.score_samples(activations)

    meta_path = os.path.join(output_dir, "metadata_" + train_opponent + ".csv")
    metadata.to_csv(meta_path, index=False)
    # print(n_components, type)
    # print('mean log prob train data', model_obj.score_samples(train_data).mean())
    # print('mean log prob validation data', model_obj.score_samples(train_opponent_validation_data).mean())

def load_metadata(env):

    metadata_path = '../activations/gmm/' + env + '/metadata_ucb.csv'
    df = pd.read_csv(metadata_path)

    # We want to evaluate on both the train and test set for the train opponent.
    # To disambiguate, we'll change the opponent_id for the train opponent in the test set.
    # For all other opponents, doesn't matter if we evaluate on "train" or "test" set
    # as they were trained on neither; we use the test set.
    is_train_opponent = df["opponent_id"] == TRAIN_ID
    # Rewrite opponent_id for train opponent in test set
    df.loc[is_train_opponent & ~df["is_train"], "opponent_id"] = TRAIN_ID + "_test"
    # Discard all non-test data, except for train opponent
    df = df.loc[is_train_opponent | ~df["is_train"]]
    return df


def bar_chart(envs, savefile=None):
    """Bar chart of mean log probability for all opponent types, grouped by environment.
    For unspecified parameters, see get_full_directory.
    :param envs: (list of str) list of environments.
    :param savefile: (None or str) path to save figure to.
    """
    dfs = []
    for env in envs:
        df = load_metadata(env)
        df["Environment"] = PRETTY_ENVS.get(env, env)
        dfs.append(df)
    longform = pd.concat(dfs)
    longform["opponent_id"] = longform["opponent_id"].apply(PRETTY_OPPONENTS.get)
    longform = longform.reset_index(drop=True)

    width, height = plt.rcParams.get("figure.figsize")
    legend_height = 0.4
    left_margin_in = 0.55
    top_margin_in = legend_height + 0.05
    bottom_margin_in = 0.5
    gridspec_kw = {
        "left": left_margin_in / width,
        "top": 1 - (top_margin_in / height),
        "bottom": bottom_margin_in / height,
    }
    fig, ax = plt.subplots(1, 1, gridspec_kw=gridspec_kw)

    # Make colors consistent with previous figures
    palette = {}
    # palette['Our'] = 'red'
    # palette['Ucb'] = '#0165FC'
    # palette['ZooT'] = '#2ca02c'
    # palette['ZooV'] = '#ff7f0e'

    palette['Our'] = 'red'
    palette['UcbT'] = '#0165FC'
    palette['Zoo'] = '#2ca02c'
    palette['UcbV'] = '#ff7f0e'

    # Actually plot
    sns.barplot(
        x="Environment",
        y="log_proba",
        hue="opponent_id",
        order=PRETTY_ENVS.values(),
        hue_order=BAR_ORDER,
        data=longform,
        palette=palette,
        errwidth=1,
    )
    ax.set_ylabel("Mean Log Probability Density")
    plt.locator_params(axis="y", nbins=4)

    # Plot our own legend
    ax.get_legend().remove()
    legend_entries = ax.get_legend_handles_labels()
    outside_legend(
        legend_entries, 3, fig, ax, ax, legend_padding=0.05, legend_height=0.6, handletextpad=0.2
    )

    if savefile is not None:
        fig.savefig(savefile)

    return fig


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default='../activations/gmm/SumoHumans')
    parser.add_argument("--output_dir", type=str, default='../activations/gmm/SumoHumans')

    args = parser.parse_args()

    activation_paths = {}

    activation_paths['norm'] = args.dir + '/activations_norm.npy'
    activation_paths['our'] = args.dir + '/activations_our_adv.npy'
    activation_paths['ucb'] = args.dir + '/activations_ucb_adv.npy'

    density_fitter(activation_paths, args.output_dir, 'ucb', n_components=25, type='full')

    styles = ["paper", "density_twocol"]
    sns.set_style("whitegrid")
    for style in styles:
        plt.style.use(STYLES[style])
    bar_chart(ENV_NAMES, savefile='../activations/gmm/chart.png')
