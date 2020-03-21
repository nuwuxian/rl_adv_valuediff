from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from pysc2.env import sc2_env

# from envs.spaces.pysc2_raw import PySC2RawAction
# from envs.spaces.pysc2_raw import PySC2RawObservation
from utils.utils import tprint

DIFFICULTIES= {
    "1": sc2_env.Difficulty.very_easy,
    "2": sc2_env.Difficulty.easy,
    "3": sc2_env.Difficulty.medium,
    "4": sc2_env.Difficulty.medium_hard,
    "5": sc2_env.Difficulty.hard,
    "6": sc2_env.Difficulty.hard,
    "7": sc2_env.Difficulty.very_hard,
    "8": sc2_env.Difficulty.cheat_vision,
    "9": sc2_env.Difficulty.cheat_money,
    "A": sc2_env.Difficulty.cheat_insane,
}


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


class SC2RawEnv(gym.Env):
    def __init__(self,
                 map_name,
                 step_mul=8,
                 resolution=32,
                 disable_fog=False,
                 agent_race='random',
                 bot_race='random',
                 difficulty='1',
                 game_steps_per_episode=None,
                 tie_to_lose=False,
                 score_index=None,
                 random_seed=None):

        """
        :param map_name: name of the used map.
        :param step_mul: How many game steps per agent step (action/observation). None means use the map default.
        :param resolution: the resolution of the agent interface. used in agent_interface_format which is passed to sc2_env.
        :param disable_fog: Whether to disable fog of war.
        :param agent_race: the race of the controlled (adversarial) agent. used for player which is passed to sc2_env.
        :param bot_race: the race of the bot (victim) agent.
        :param difficulty: the difficulty of the bot agent.
        :param game_steps_per_episode: Game steps per episode, independent of the step_mul. 0 means no limit.
                                       None means use the map default.
        :param tie_to_lose: count tie games as losses.
        :param score_index: -1 means use the win/loss reward, >=0 is the index into the score_cumulative with
                            0 being the curriculum score. None means use the map default.
        :param random_seed: Random number seed to use when initializing the game.
        variable members: action_space, observation_space.
        function members: step, reset, close.
        """
        self._map_name = map_name
        self._step_mul = step_mul
        self._resolution = resolution
        self._disable_fog = disable_fog
        self._agent_race = agent_race
        self._bot_race = bot_race
        self._difficulty = difficulty
        self._game_steps_per_episode = game_steps_per_episode
        self._tie_to_lose = tie_to_lose
        self._score_index = score_index
        self._random_seed = random_seed
        self._reseted = False
        self._first_create = True

        # create game environment.
        self._sc2_env = self._safe_create_env()
        # get observation space.
        self.observation_space = PySC2RawObservation(self._sc2_env.observation_spec)
        # get action space.
        self.action_space = PySC2RawAction()

    def step(self, actions):
        """
        :param actions: agent action at the t.
        :return: observation t+1, reward t, done t, info t.
        """
        # todo: get the reward for another agent and add somethings in info for computing winning rate.
        # add computing winning rate
        assert self._reseted
        timestep = self._sc2_env.step([actions])[0]
        observation = timestep.observation
        reward = float(timestep.reward)
        done = timestep.last()
        info = {}
        if done:
          self._reseted = False
          if self._tie_to_lose and reward == 0:
            reward = -1.0
          tprint("Episode Done. Difficulty: %s Outcome %f" %
                 (self._difficulty, reward))
          if reward > 0:
              info['winning'] = True
              info['tie'] = False
              info['loss'] = False
          elif reward == 0:
              info['winning'] = False
              info['tie'] = True
              info['loss'] = False
          else:
              info['winning'] = False
              info['tie'] = False
              info['loss'] = True
        return (observation, reward, done, info)

    def reset(self):
        """ reset game environment.
        :return: observation at t0.
        """
        timesteps = self._safe_reset()
        self._reseted = True
        return timesteps[0].observation

    def _reset(self):
        """ true reset game environment.
        :return: sc2_env.reset().
        """
        if not self._first_create:
            self._sc2_env.close()
            self._sc2_env = self._create_env()
            self._first_create = False
        return self._sc2_env.reset()

    def _safe_reset(self, max_retry=10):
        for _ in range(max_retry - 1):
            try: return self._reset()
            except: pass
        return self._reset()

    def close(self):
        self._sc2_env.close()

    def _create_env(self):
        """ create game environment.
        :return: sc2_env.SC2Env().
        """
        self._random_seed = (self._random_seed + 11) & 0xFFFFFFFF
        players=[sc2_env.Agent(sc2_env.Race[self._agent_race]),
                 sc2_env.Bot(sc2_env.Race[self._bot_race],
                              DIFFICULTIES[self._difficulty])]
        agent_interface_format=sc2_env.parse_agent_interface_format(
            feature_screen=self._resolution, feature_minimap=self._resolution)
        tprint("Creating game with seed %d." % self._random_seed)
        return sc2_env.SC2Env(
            map_name=self._map_name,
            step_mul=self._step_mul,
            players=players,
            agent_interface_format=agent_interface_format,
            disable_fog=self._disable_fog,
            game_steps_per_episode=self._game_steps_per_episode,
            visualize=False,
            score_index=self._score_index,
            random_seed=self._random_seed)

    def _safe_create_env(self, max_retry=10):
        for _ in range(max_retry - 1):
          try: return self._create_env()
          except: pass
        return self._create_env()
