import numpy as np
import tensorflow as tf
from tensorflow.python import keras
import h5py
import os

# define the generator
class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_sz, dim, n_class, shuffle, len, in_dir):
        self.batch_sz = batch_sz
        self.dim = dim
        self.n_class = n_class
        self.shuffle = shuffle
        self.feats = h5py.File(os.path.join(in_dir, 'feats.h5'), 'r')
        self.labels = h5py.File(os.path.join(in_dir, 'labels.h5'), 'r')
        self.len = len
        self.on_epoch_end()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        idx = self.indexes[index * self.batch_sz: (index + 1) * self.batch_sz]
        data, label = self.__data_generation(idx)
        return data, label

    def on_epoch_end(self):
        self.idx = np.arrange(len(self.len))
        if self.shuffle == True:
            np.random.shuffle(self.idx)

    def __data_generation(self, idx):

        input = np.zeros((self.batch_sz, self.dim))
        output = np.zeros((self.batch_sz, self.n_class))

        for i, id in enumerate(idx):
            input[i,:] = self.feats[str(id)]
            output[i,:] = self.labels[str(id)]
        return input, output
