from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym


class PySC2RawAction(gym.Space):
    """ gym action space wrapper.
    """
    pass


class PySC2RawObservation(gym.Space):
    """ gym observation space wrapper.
    """

    def __init__(self, observation_spec_fn):
        self._feature_layers = observation_spec_fn()

    @property
    def space_attr(self):
        return self._feature_layers
