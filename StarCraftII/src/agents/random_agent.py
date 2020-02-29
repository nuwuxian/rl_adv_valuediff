from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from envs.spaces.mask_discrete import MaskDiscrete
from envs.spaces.pysc2_raw import PySC2RawAction


class RandomAgent(object):
    def __init__(self, action_space):
        """ Random agent.
        :param action_space: the dimensionality of the agent's action.
        """
        self._action_space = action_space

    def act(self, observation, eps=0):
        """ receive an observation and take an action.
        :param observation: the input observation.
        :param eps: ..
        :return: the output action.
        """
        if (isinstance(self._action_space, MaskDiscrete) or isinstance(self._action_space, PySC2RawAction)):
            action_mask = observation[-1]
            return self._action_space.sample(np.nonzero(action_mask)[0])
        else:
            return self._action_space.sample()

    def reset(self):
        pass
