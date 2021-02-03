import numpy as np
import argparse
import os
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib


def fit_tsne(activation_paths, output_dir):

    sub_data = []
    
    for opponent_type, path in activation_paths.items():
        file_data = np.load(path, allow_pickle=True)
        sub_data.append(file_data)

    sub_data = np.concatenate(sub_data)
 
    # Fit t-SNE
    tsne_obj = TSNE(n_components=2, verbose=1, perplexity=250)
    tsne_ids = tsne_obj.fit_transform(sub_data)

    # Save weights
    tsne_weights_path = os.path.join(output_dir, "tsne_weights.pkl")
    with open(tsne_weights_path, "wb") as fp:
        pickle.dump(tsne_obj, fp)

    # Save cluster IDs
    cluster_ids_path = os.path.join(output_dir, "cluster_ids.npy")
    np.save(cluster_ids_path, tsne_ids)



def plot_graph(cluster_ids_path, output_dir):


    data = np.load(cluster_ids_path)
    # plot the graph
    fig, ax = plt.subplots(figsize=(3.2, 3.2))

    colors = {"Our": "orangered", "Zoo": '#2ca02c', "Ucb": "blue"}
    len = int(data.shape[0] / 3)

    opponent_type = ['Zoo'] * len + ['Our'] * len + ['Ucb'] * len

    metadata_df = pd.DataFrame(
        {
            "opponent_id": opponent_type,
        }
    )

    hues = metadata_df["opponent_id"].apply(colors.get)

    ax.axis("off")
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.margins(x=0.01, y=0.01)
    ax.scatter(data[:, 0], data[:, 1], c=hues, alpha=0.75, s=0.25, edgecolors="none", linewidth=1)
    fig.savefig(output_dir + '/' + output_dir.split('/')[-1] + '.pdf')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default='../activations/tsne/YouShallNotPassHumans')
    parser.add_argument("--output_dir", type=str, default='../activations/tsne/YouShallNotPassHumans')

    args = parser.parse_args()

    activation_paths = {}
    activation_paths['norm'] = args.dir + '/activations_norm.npy'
    activation_paths['our'] = args.dir + '/activations_our_adv.npy'
    activation_paths['ucb'] = args.dir + '/activations_ucb_adv.npy'

    fit_tsne(activation_paths, args.output_dir)
    cluster_ids_path = args.dir + '/cluster_ids.npy'
    plot_graph(cluster_ids_path, args.output_dir)
