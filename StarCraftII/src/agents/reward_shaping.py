from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pysc2.lib.typeenums import UNIT_TYPEID as UNIT_TYPE
from envs.common.const import ALLY_TYPE

reward_range = (-np.inf, np.inf)


def RewardShapingV1(units_t, units_t_1, reward, done):
    """
    reward shaping 1: game reward * 10 + num_iteration that n_combats > n_enemies.
    """
    def _get_unit_counts(units):
        _combat_unit_types = set([UNIT_TYPE.ZERG_ZERGLING.value, UNIT_TYPE.ZERG_ROACH.value,
                                  UNIT_TYPE.ZERG_HYDRALISK.value])
        num_enemy_units, num_self_combat_units = 0, 0
        for u in units:
            if u.int_attr.alliance == ALLY_TYPE.ENEMY.value:
                num_enemy_units += 1
            elif u.int_attr.alliance == ALLY_TYPE.SELF.value:
                if u.unit_type in _combat_unit_types:
                    num_self_combat_units += 1
        return num_enemy_units, num_self_combat_units

    n_enemies_t, n_self_combats_t = _get_unit_counts(units_t)
    n_enemies_t_1, n_self_combats_t_1 = _get_unit_counts(units_t_1)

    if n_self_combats_t - n_enemies_t > n_self_combats_t_1 - n_enemies_t_1:
        reward_t = 1
    elif n_self_combats_t - n_enemies_t > n_self_combats_t_1 - n_enemies_t_1:
        reward_t = -1
    else:
        reward_t = 0
    if not done:
        reward = reward * 10 + reward_t
    else:
        reward = reward * 10
    return reward


def RewardShapingV2(units_t, units_t_1, reward, done):
    """
    reward shaping 2: game reward  + num_selves at each iteration * 0.02.
    """
    def _get_unit_counts(units):
        _combat_unit_types = set([UNIT_TYPE.ZERG_ZERGLING.value,
                                  UNIT_TYPE.ZERG_ROACH.value,
                                  UNIT_TYPE.ZERG_HYDRALISK.value,
                                  UNIT_TYPE.ZERG_RAVAGER.value,
                                  UNIT_TYPE.ZERG_BANELING.value,
                                  UNIT_TYPE.ZERG_BROODLING.value])
        num_enemy_units, num_self_units = 0, 0
        for u in units:
            if u.int_attr.alliance == ALLY_TYPE.ENEMY.value:
                if u.unit_type in _combat_unit_types:
                    num_enemy_units += 1
            elif u.int_attr.alliance == ALLY_TYPE.SELF.value:
                if u.unit_type in _combat_unit_types:
                    num_self_units += 1
        return num_enemy_units, num_self_units

    n_enemies_t, n_selves_t = _get_unit_counts(units_t)
    n_enemies_t_1, n_selves_t_1 = _get_unit_counts(units_t_1)

    diff_selves = n_selves_t - n_selves_t_1
    diff_enemies = n_enemies_t - n_enemies_t_1
    if not done:
        reward += (diff_selves - diff_enemies) * 0.02
    return reward


def KillingReward(kill_t, kill_t_1, reward, done):

    """
    reward shaping 3: game reward  + num_kills at each iteration * 1e-5.
    """
    if not done:
        reward += (kill_t - kill_t_1) * 1e-5
    return reward