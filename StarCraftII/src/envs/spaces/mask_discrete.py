from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from gym.spaces.discrete import Discrete


class MaskDiscrete(Discrete):
    """A discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`. __init__ param: n.
    """

    def sample(self, availables):
        """ randomly sampling one item from the availables
        :param availables: available choices.
        :return: the sampled choice.
        """
        x = np.random.choice(availables).item()
        assert self.contains(x, availables)
        return x

    def contains(self, x, availables):
        """
        :param x: where x in the value range n and in availables.
        :param availables: available choices.
        :return: Bool.
        """
        return super(MaskDiscrete, self).contains(x) and x in availables

    def __repr__(self):
        return "MaskDiscrete(%d)" % self.n
