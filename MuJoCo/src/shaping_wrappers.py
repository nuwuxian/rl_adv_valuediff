from collections import deque
from itertools import islice

from stable_baselines.common.vec_env import VecEnvWrapper
from scheduling import ConditionalAnnealer, ConstantAnnealer, LinearAnnealer

REW_TYPES = set(('sparse', 'dense'))


class RewardShapingVecWrapper(VecEnvWrapper):
    """
    A more direct interface for shaping the reward of the attacking agent.
    - shaping_params schema: {'sparse': {k: v}, 'dense': {k: v}, **kwargs}
    """
    def __init__(self, venv, agent_idx, shaping_params, total_step, reward_annealer=None):
        """
        :param: venv: environment.
        :param: shaping_params: shaping parameters (dense and sparse reward coefficient).
        :param: agent_idx: agent index.
        :param: scheduler: reward_annealer: anneal coefficient.
        """
        super().__init__(venv)
        assert shaping_params.keys() == REW_TYPES
        self.shaping_params = {}
        for rew_type, params in shaping_params.items():
            for rew_term, weight in params.items():
                self.shaping_params[rew_term] = (rew_type, weight)

        self.reward_annealer = reward_annealer
        self.agent_idx = agent_idx
        queue_keys = REW_TYPES.union(['length'])
        self.ep_logs = {k: deque([], maxlen=10000) for k in queue_keys}
        self.ep_logs['total_episodes'] = 0
        self.ep_logs['last_callback_episode'] = 0
        self.step_rew_dict = {rew_type: [[] for _ in range(self.num_envs)]
                              for rew_type in REW_TYPES}
        self.total_step = total_step
        self.cnt = 0

    def get_logs(self):
        """Interface to access self.ep_logs which contains data about episodes"""
        if self.ep_logs['total_episodes'] == 0:
            return None
        # keys: 'dense', 'sparse', 'length', 'total_episodes', 'last_callback_episode'
        return self.ep_logs

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        self.cnt = self.cnt + 20
        frac_remaining = max(1 - self.cnt/self.total_step, 0)
        obs, rew, done, infos = self.venv.step_wait()
        for env_num in range(self.num_envs):
            # Compute shaped_reward for each rew_type
            shaped_reward = {k: 0 for k in REW_TYPES}
            for rew_term, rew_value in infos[env_num].items():
                if rew_term not in self.shaping_params:
                    continue
                rew_type, weight = self.shaping_params[rew_term]
                shaped_reward[rew_type] += weight * rew_value
            # Compute total shaped reward, optionally annealing
            rew[env_num] = _anneal(shaped_reward, self.reward_annealer, frac_remaining)
            # Log the results of an episode into buffers and then pass on the shaped reward
            for rew_type, val in shaped_reward.items():
                self.step_rew_dict[rew_type][env_num].append(val)

            if done[env_num]:
                ep_length = max(len(self.step_rew_dict[k]) for k in self.step_rew_dict.keys())
                self.ep_logs['length'].appendleft(ep_length)
                for rew_type in REW_TYPES:
                    rew_type_total = sum(self.step_rew_dict[rew_type][env_num])
                    self.ep_logs[rew_type].appendleft(rew_type_total)
                    self.step_rew_dict[rew_type][env_num] = []
                self.ep_logs['total_episodes'] += 1
        return obs, rew, done, infos


def apply_reward_wrapper(single_env, shaping_params, agent_idx, scheduler, total_step):
    """ reward shaping wrapper.
    :param: single_env: environment.
    :param: shaping_params: shaping parameters.
    :param: agent_idx: agent index.
    :param: scheduler: anneal factor scheduler.
    """
    if 'metric' in shaping_params:
        rew_shape_annealer = ConditionalAnnealer.from_dict(shaping_params, get_logs=None)
        scheduler.set_conditional('rew_shape')
    else:
        anneal_frac = shaping_params.get('anneal_frac')
        if shaping_params.get('anneal_type')==0:
            rew_shape_annealer = ConstantAnnealer(anneal_frac)
        else:
            rew_shape_annealer = LinearAnnealer(1, 0, anneal_frac)

    scheduler.set_annealer('rew_shape', rew_shape_annealer)
    return RewardShapingVecWrapper(single_env, agent_idx=agent_idx,
                                   shaping_params=shaping_params['weights'],
                                   reward_annealer=scheduler.get_annealer('rew_shape'),
                                   total_step=total_step)


def _anneal(reward_dict, reward_annealer, frac_remaining):
    c = reward_annealer(frac_remaining)
    #print(c)
    #print('===============================')
    assert 0 <= c <= 1

    sparse_weight = 1 - c
    dense_weight = c

    return (reward_dict['sparse'] * sparse_weight
            + reward_dict['dense'] * dense_weight)
