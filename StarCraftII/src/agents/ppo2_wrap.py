from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from collections import deque
from queue import Queue
import queue
from threading import Thread
import time
import random
import joblib

import numpy as np
import tensorflow as tf
import zmq
from gym import spaces

from envs.spaces.mask_discrete import MaskDiscrete
from utils.utils import tprint
from agents.ppo_agent import Model
from agents.utils_tf import explained_variance
from agents.reward_shaping import RewardShapingV1, RewardShapingV2, KillingReward

class Adv_Model(object):
    """
    PPO objective function and training.
    """
    def __init__(self, *, policy, value, ob_space, ac_space, nbatch_act, nbatch_train, vf_coef,
                 unroll_length, ent_coef, max_grad_norm, scope_name, value_clip=False):
        """
        :param policy: adversarial policy network and value function.
        :param value: victim agent value function and diff value function.
        :param ob_space: observation space.
        :param ac_space: action space.
        :param nbatch_act: act model batch size.
        :param nbatch_train: training model batch size.
        :param vf_coef: value function loss coefficient.
        :param unroll_length: training rollout length, used for lstm.
        :param ent_coef: entropy loss coefficient.
        :param max_grad_norm: gradient clip.
        :param scope_name: model scope.
        :param value_clip: return clip or not.
        """
        sess = tf.get_default_session()

        act_model = policy(sess, scope_name, ob_space, ac_space, nbatch_act, 1,
                           reuse=False)
        train_model = policy(sess, scope_name, ob_space, ac_space, nbatch_train,
                             unroll_length, reuse=True)

        # value_model:  opponent agent value function model.
        # value1_model: diff value function model.

        vact_model = value(sess, "oppo_value", ob_space, ac_space, nbatch_act, 1,
                           reuse=False)
        vtrain_model = value(sess, "oppo_value", ob_space, ac_space, nbatch_train,
                             unroll_length, reuse=True)

        vact1_model = value(sess, "diff_value", ob_space, ac_space, nbatch_act, 1,
                            reuse=False)
        vtrain1_model = value(sess, "diff_value", ob_space, ac_space, nbatch_train,
                              unroll_length, reuse=True)

        A = tf.placeholder(shape=(nbatch_train,), dtype=tf.int32)
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        self.coef_opp_ph = tf.placeholder(tf.float32, [], name="coef_opp_ph")
        self.coef_adv_ph = tf.placeholder(tf.float32, [], name="coef_adv_ph")
        self.coef_abs_ph = tf.placeholder(tf.float32, [], name="coef_abs_ph")

        opp_ADV = tf.placeholder(tf.float32, [None])
        opp_R = tf.placeholder(tf.float32, [None])
        opp_OLDVPRED = tf.placeholder(tf.float32, [None])

        abs_ADV = tf.placeholder(tf.float32, [None])
        abs_R = tf.placeholder(tf.float32, [None])
        abs_OLDVPRED = tf.placeholder(tf.float32, [None])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED,
                                                   -CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        if value_clip:
          vf_losses2 = tf.square(vpredclipped - R)
          vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        else:
          vf_loss = .5 * tf.reduce_mean(vf_losses1)

        # opp value_loss
        opp_vpred = vtrain_model.vf
        opp_vpredclipped = opp_OLDVPRED + tf.clip_by_value(vtrain_model.vf - opp_OLDVPRED, -CLIPRANGE, CLIPRANGE)
        opp_vf_losses1 = tf.square(opp_vpred - opp_R)
        if value_clip:
          opp_vf_losses2 = tf.square(opp_vpredclipped - opp_R)
          opp_vf_loss = .5 * tf.reduce_mean(tf.maximum(opp_vf_losses1, opp_vf_losses2))
        else:
          opp_vf_loss = .5 * tf.reduce_mean(opp_vf_losses1)

        # diff value loss
        diff_vpred = vtrain1_model.vf
        diff_vpredclipped = abs_OLDVPRED + tf.clip_by_value(vtrain1_model.vf - abs_OLDVPRED, -CLIPRANGE, CLIPRANGE)
        diff_vf_losses1 = tf.square(diff_vpred - abs_R)
        if value_clip:
          diff_vf_losses2 = tf.square(diff_vpredclipped - abs_R)
          diff_vf_loss = .5 * tf.reduce_mean(tf.maximum(diff_vf_losses1, diff_vf_losses2))
        else:
          diff_vf_loss = .5 * tf.reduce_mean(diff_vf_losses1)

        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = (self.coef_abs_ph * abs_ADV + self.coef_opp_ph * opp_ADV
                     + self.coef_adv_ph * ADV) * ratio
        pg_losses2 = (self.coef_abs_ph * abs_ADV + self.coef_opp_ph * opp_ADV
                     + self.coef_adv_ph * ADV) * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))

        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        # total loss, add value function
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + \
               opp_vf_loss * vf_coef + diff_vf_loss * vf_coef

        params = tf.trainable_variables(scope=scope_name)
        params += tf.trainable_variables(scope="oppo_value")
        params += tf.trainable_variables(scope="diff_value")

        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
          grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)
        new_params = [tf.placeholder(p.dtype, shape=p.get_shape()) for p in params]
        param_assign_ops = [p.assign(new_p) for p, new_p in zip(params, new_params)]

        def train(lr, cliprange, coef_opp, coef_adv, coef_abs, obs, returns, dones, actions, values,
                  neglogpacs,  opp_obs, opp_returns, opp_values, abs_returns, abs_values,
                  states=None, opp_states=None, abs_states=None):

            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            opp_advs = opp_returns - opp_values
            opp_advs = (opp_advs - opp_advs.mean()) / (opp_advs.std() + 1e-8)

            abs_advs = abs_returns - abs_values
            abs_advs = (abs_advs - abs_advs.mean()) / (abs_advs.std() + 1e-8)

            if isinstance(ac_space, MaskDiscrete):
                td_map = {train_model.X: obs[0], train_model.MASK: obs[-1], A: actions,
                          ADV: advs, R:returns, LR:lr, CLIPRANGE:cliprange,
                          self.coef_opp_ph:coef_opp, self.coef_adv_ph:coef_adv, self.coef_abs_ph:coef_abs,
                          OLDNEGLOGPAC:neglogpacs, OLDVPRED:values,
                          vtrain_model.X:opp_obs[0], vtrain_model.MASK:opp_obs[-1],
                          opp_ADV:opp_advs, opp_R:opp_returns, opp_OLDVPRED:opp_values,
                          vtrain1_model.X:opp_obs[0], vtrain1_model.MASK:opp_obs[-1],
                          abs_ADV:abs_advs, abs_R:abs_returns, abs_OLDVPRED:abs_values}
            else:
                td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                          CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values,
                          self.coef_opp_ph: coef_opp, self.coef_adv_ph: coef_adv, self.coef_abs_ph: coef_abs,
                          opp_ADV:opp_advs, opp_R:opp_returns, opp_OLDVPRED:opp_values,
                          abs_ADV:abs_advs, abs_R:abs_returns, abs_OLDVPRED:abs_values}

            if states is not None:
                td_map[train_model.STATE] = states
                td_map[train_model.DONE] = dones

            if opp_states is not None:
                td_map[vtrain_model.STATE] = opp_states
                td_map[vtrain_model.DONE] = dones

            if abs_states is not None:
                td_map[vtrain1_model.STATE] = abs_states
                td_map[vtrain1_model.DONE] = dones

            return sess.run([pg_loss, vf_loss, entropy, approxkl, clipfrac, _train], td_map)[:-1]

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy',
                           'approxkl', 'clipfrac']

        def save(save_path):
            joblib.dump(read_params(), save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            load_params(loaded_params)

        def read_params():
            return sess.run(params)

        def load_params(loaded_params):
            sess.run(param_assign_ops[0:len(loaded_params)], feed_dict={p : v for p, v in zip(new_params[0:len(loaded_params)], loaded_params)})

        self.train = train
        self.train_model = train_model
        self.act_model = act_model

        self.vtrain_model = vtrain_model
        self.vact_model = vact_model

        self.vtrain1_model = vtrain1_model
        self.vact1_model = vact1_model

        self.step = act_model.step
        self.value = act_model.value

        self.opp_value = vact_model.value
        self.abs_value = vact1_model.value

        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        self.read_params = read_params
        self.load_params = load_params

        tf.global_variables_initializer().run(session=sess)


class PPO_AdvActor(object):
    """
    actor or runner.
    """
    def __init__(self, env, policy, value, unroll_length, gamma, lam, queue_size=1,
                 enable_push=True, learner_ip="localhost", port_A="5700", port_B="5701",
                 reward_shape='none', use_victim_ob=False, victim_model=None):
        """
        :param env: environment, selfplay.
        :param policy: adversarial policy network and value function.
        :param value: victim agent value function and diff value function.
        :param unroll_length: n_batch training and training rollout length, used for lstm.
        :param gamma: discount factor.
        :param lam: used to compute
        :param queue_size: use for communicate with learner.
        :param enable_push: use for communicate with learner.
        :param learner_ip: ip of learner.
        :param port_A:
        :param port_B:
        :param reward_shape: type of reward shaping.
        :param use_victim_ob: use victim agent's observation or not.
        :param victim_model: victim policy model path.
        """

        self._env = env
        self._unroll_length = unroll_length
        self._lam = lam
        self._gamma = gamma
        self._enable_push = enable_push
        self.reward_shaping = reward_shape

        self.use_victim_ob = use_victim_ob

        self._model = Adv_Model(policy=policy, value=value,
                                scope_name="model",
                                ob_space=env.observation_space,
                                ac_space=env.action_space,
                                nbatch_act=1,
                                nbatch_train=unroll_length,
                                unroll_length=unroll_length,
                                ent_coef=0.01,
                                vf_coef=0.5,
                                max_grad_norm=0.5)

        self._oppo_model = Model(policy=policy,
                                 scope_name="oppo_model",
                                 ob_space=env.observation_space,
                                 ac_space=env.action_space,
                                 nbatch_act=1,
                                 nbatch_train=unroll_length,
                                 unroll_length=unroll_length,
                                 ent_coef=0.01,
                                 vf_coef=0.5,
                                 max_grad_norm=0.5)

        # use zip, still need double check. load victim model, check this part.
        if victim_model != None:
            self._oppo_model.load(victim_model)

        self._obs, self._oppo_obs = env.reset()
        # define the state and oppo_state
        self._state = self._model.initial_state
        self._oppo_state = self._oppo_model.initial_state

        # init_states for adversary
        self.adv_opp_states = self._model.initial_state
        self.adv_abs_states = self._model.initial_state

        self._done = False
        self._cum_reward = 0

        self._zmq_context = zmq.Context()
        self._model_requestor = self._zmq_context.socket(zmq.REQ)
        self._model_requestor.connect("tcp://%s:%s" % (learner_ip, port_A))
        if enable_push:
            self._data_queue = Queue(queue_size)
            self._push_thread = Thread(target=self._push_data, args=(
                self._zmq_context, learner_ip, port_B, self._data_queue))
            self._push_thread.start()

    def run(self):
        while True:
            # fetch model
            t = time.time()
            self._update_model()
            tprint("Update model time: %f" % (time.time() - t))
            t = time.time()
            # rollout, batch_size: unroll_length
            unroll = self._nstep_rollout()
            if self._enable_push:
                if self._data_queue.full(): tprint("[WARN]: Actor's queue is full.")
                self._data_queue.put(unroll)
                tprint("Rollout time: %f" % (time.time() - t))

    def _nstep_rollout(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]

        # define the opponent observation, rewards, dones
        mb_opp_obs, mb_opp_returns, mb_opp_values, mb_abs_returns, mb_abs_values = [],[],[],[],[]

        mb_opp_rewards, mb_abs_rewards = [], []

        mb_states, episode_infos = self._state, []
        mb_adv_opp_states = self.adv_opp_states
        mb_adv_abs_states = self.adv_abs_states

        # two multi-agent competition
        # add opponent actions and values
        units_t_1 = []
        units_oppo_t_1 = []
        kill_t_1 = 0
        kill_oppo_t_1 = 0

        for _ in range(self._unroll_length):
            action, value, self._state, neglogpac = self._model.step(
                transform_tuple(self._obs, lambda x: np.expand_dims(x, 0)),
                self._state,
                np.expand_dims(self._done, 0))

            oppo_action, _, self._oppo_state, _ = self._oppo_model.step(
                transform_tuple(self._oppo_obs, lambda x: np.expand_dims(x, 0)),
                self._oppo_state,
                np.expand_dims(self._done, 0))

            mb_obs.append(transform_tuple(self._obs, lambda x: x.copy()))
            mb_actions.append(action[0])
            mb_values.append(value[0])
            mb_neglogpacs.append(neglogpac[0])
            mb_dones.append(self._done)

            if self.use_victim_ob:
                mb_opp_obs.append(transform_tuple(self._oppo_obs, lambda x: x.copy()))
            else:
                mb_opp_obs.append(transform_tuple(self._obs, lambda x: x.copy()))

            if self.use_victim_ob:
                obs_oppo = self._oppo_obs
            else:
                obs_oppo = self._obs

            values_oppo, self.adv_opp_states = self._model.opp_value(transform_tuple(obs_oppo,
                                                                                     lambda x: np.expand_dims(x, 0)),
                                                                     self.adv_opp_states,
                                                                     np.expand_dims(self._done, 0))

            values_abs, self.adv_abs_states = self._model.abs_value(transform_tuple(obs_oppo,
                       lambda x: np.expand_dims(x, 0)), self.adv_abs_states,
                       np.expand_dims(self._done, 0))

            mb_opp_values.append(values_oppo[0])
            mb_abs_values.append(values_abs[0])

            (self._obs, self._oppo_obs), reward, self._done, info \
                = self._env.step([action[0], oppo_action[0]])

            units_t = info['units'][0]
            units_oppo_t = info['units'][1]
            kill_t = info['killing'][0]
            kill_oppo_t = info['killing'][1]

            # reward shaping.
            if self.reward_shaping == 'kill':
                reward = KillingReward(kill_t, kill_t_1, reward, self._done)
                oppo_reward = KillingReward(kill_oppo_t, kill_oppo_t_1, info['oppo_reward'], self._done)
            elif self.reward_shaping == 'v1':
                reward = RewardShapingV1(units_t, units_t_1, reward, self._done)
                oppo_reward = RewardShapingV1(units_oppo_t, units_oppo_t_1, info['oppo_reward'], self._done)
            elif self.reward_shaping == 'v2':
                reward = RewardShapingV1(units_t, units_t_1, reward, self._done)
                oppo_reward = RewardShapingV1(units_oppo_t, units_oppo_t_1, info['oppo_reward'], self._done)
            else:
                oppo_reward = info['oppo_reward']

            units_t_1 = units_t
            units_oppo_t_1 = units_oppo_t
            kill_t_1 = kill_t
            kill_oppo_t_1 = kill_oppo_t

            self._cum_reward += reward
            if self._done:
                self._obs, self._oppo_obs = self._env.reset()
                self._state = self._model.initial_state
                self._oppo_state = self._oppo_model.initial_state
                self.adv_opp_states = self._model.initial_state
                self.adv_abs_states = self._model.initial_state
                episode_infos.append({'r': self._cum_reward, 'win': int(info['winning']), 'tie': int(info['tie']),
                                      'loss': int(info['loss']),})
                self._cum_reward = 0

            # opp, abs rewards
            mb_rewards.append(reward)
            mb_opp_rewards.append(oppo_reward)
            mb_abs_rewards.append((reward - oppo_reward))

        if isinstance(self._obs, tuple):
            mb_obs = tuple(np.asarray(obs, dtype=self._obs[0].dtype)
                         for obs in zip(*mb_obs))
            mb_opp_obs = tuple(np.asarray(obs, dtype=self._obs[0].dtype)
                         for obs in zip(*mb_opp_obs))
        else:
            mb_obs = np.asarray(mb_obs, dtype=self._obs.dtype)
            mb_opp_obs = np.asarray(mb_opp_obs, dtype=self._obs.dtype)

        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)

        # Add abs-rewards, abs-dones
        mb_opp_rewards = np.asarray(mb_opp_rewards, dtype=np.float32)
        mb_opp_values = np.asarray(mb_opp_values, dtype=np.float32)

        mb_abs_rewards = np.asarray(mb_abs_rewards, dtype=np.float32)
        mb_abs_values = np.asarray(mb_abs_values, dtype=np.float32)
        last_values = self._model.value(
            transform_tuple(self._obs, lambda x: np.expand_dims(x, 0)),
            self._state,
            np.expand_dims(self._done, 0))

        if self.use_victim_ob:
          opp_last_values, _ = self._model.opp_value(transform_tuple(self._oppo_obs,
                       lambda x: np.expand_dims(x, 0)), self.adv_opp_states)
          abs_last_values, _ = self._model.abs_value(transform_tuple(self._oppo_obs,
                       lambda x: np.expand_dims(x, 0)), self.adv_abs_states)
        else:
          opp_last_values, _ = self._model.opp_value(transform_tuple(self._obs,
                       lambda x: np.expand_dims(x, 0)), self.adv_opp_states)
          abs_last_values, _ = self._model.abs_value(transform_tuple(self._obs,
                       lambda x: np.expand_dims(x, 0)), self.adv_abs_states)

        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)

        mb_opp_returns = np.zeros_like(mb_opp_rewards)
        mb_opp_advs = np.zeros_like(mb_opp_rewards)

        mb_abs_returns = np.zeros_like(mb_abs_rewards)
        mb_abs_advs = np.zeros_like(mb_abs_rewards)

        last_gae_lam = 0
        opp_last_gae_lam = 0
        abs_last_gae_lam = 0

        for t in reversed(range(self._unroll_length)):
            if t == self._unroll_length - 1:
                next_nonterminal = 1.0 - self._done
                next_values = last_values[0]
                opp_nextvalues = opp_last_values[0]
                abs_nextvalues = abs_last_values[0]
            else:
                next_nonterminal = 1.0 - mb_dones[t + 1]
                next_values = mb_values[t + 1]
                opp_nextvalues = mb_opp_values[t + 1]
                abs_nextvalues = mb_abs_values[t + 1]

            delta = mb_rewards[t] + self._gamma * next_values * next_nonterminal - mb_values[t]
            mb_advs[t] = last_gae_lam = delta + self._gamma * self._lam * next_nonterminal * last_gae_lam

            # opp-delta
            opp_delta = mb_opp_rewards[t] + self._gamma * opp_nextvalues * next_nonterminal - mb_opp_values[t]
            mb_opp_advs[t] = opp_last_gae_lam = opp_delta + self._gamma * self._lam * \
                                                next_nonterminal * opp_last_gae_lam
            # abs-delta
            abs_delta = mb_abs_rewards[t] + self._gamma * abs_nextvalues * next_nonterminal - mb_abs_values[t]
            mb_abs_advs[t] = abs_last_gae_lam = abs_delta + self._gamma * self._lam * \
                                                next_nonterminal * abs_last_gae_lam

        mb_returns = mb_advs + mb_values
        mb_opp_returns = mb_opp_advs + mb_opp_values
        mb_abs_returns = mb_abs_advs + mb_abs_values

        # Shape: [unroll_length, XX]. batch_size: unroll_length
        # opp_obs, opp_returns, opp_values, abs_returns, abs_values
        # states = None, opp_states = None, abs_states = None
        return (mb_obs, mb_opp_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs,
                mb_opp_returns, mb_opp_values, mb_abs_returns, mb_abs_values,
                mb_states, mb_adv_opp_states, mb_adv_abs_states, episode_infos)

    def _push_data(self, zmq_context, learner_ip, port_B, data_queue):
        sender = zmq_context.socket(zmq.PUSH)
        sender.setsockopt(zmq.SNDHWM, 1)
        sender.setsockopt(zmq.RCVHWM, 1)
        sender.connect("tcp://%s:%s" % (learner_ip, port_B))
        while True:
            data = data_queue.get()
            sender.send_pyobj(data)

    def _update_model(self):
        self._model_requestor.send_string("request model")
        self._model.load_params(self._model_requestor.recv_pyobj())


# Adv Learner
class Adv_Learner(object):
  def __init__(self, env, policy, value, unroll_length, lr, clip_range, batch_size,
               ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, queue_size=8,
               print_interval=100, save_interval=10000, learn_act_speed_ratio=0,
               unroll_split=8, save_dir=None, init_model_path=None, max_episode=4,
               port_A="5700", port_B="5701", coef_opp_init=1, coef_opp_schedule='const',
               coef_adv_init=1,  coef_adv_schedule='const', coef_abs_init=1, coef_abs_schedule='const'):
    """
     queue_size: maximum queue size per update.
     max_episode: maximum games per update.
    """

    assert isinstance(env.action_space, spaces.Discrete)
    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(clip_range, float): clip_range = constfn(clip_range)
    else: assert callable(clip_range)
    self._lr = lr
    self._clip_range=clip_range
    self._batch_size = batch_size
    self._unroll_length = unroll_length
    self._print_interval = print_interval
    self._save_interval = save_interval
    self._learn_act_speed_ratio = learn_act_speed_ratio
    self._save_dir = save_dir

    # Set the constant
    self.coef_opp_init = coef_opp_init
    self.coef_opp_schedule = coef_opp_schedule

    self.coef_adv_init = coef_adv_init
    self.coef_adv_schedule = coef_adv_schedule

    self.coef_abs_init = coef_abs_init
    self.coef_abs_schedule = coef_abs_schedule

    self._model = Adv_Model(policy=policy, value=value,
                        scope_name="model",
                        ob_space=env.observation_space,
                        ac_space=env.action_space,
                        nbatch_act=1,
                        nbatch_train=unroll_length * batch_size,
                        unroll_length=unroll_length,
                        ent_coef=ent_coef,
                        vf_coef=vf_coef,
                        max_grad_norm=max_grad_norm)
    if init_model_path is not None: self._model.load(init_model_path)
    self._model_params = self._model.read_params()
    self._unroll_split = unroll_split if self._model.initial_state is None else 1
    assert self._unroll_length % self._unroll_split == 0
    self._data_queue = deque(maxlen=queue_size * self._unroll_split)
    self._data_timesteps = deque(maxlen=200)
    self._episode_infos = deque(maxlen=max_episode)
    self._num_unrolls = 0

    self._zmq_context = zmq.Context()
    self._pull_data_thread = Thread(
        target=self._pull_data,
        args=(self._zmq_context, self._data_queue, self._episode_infos,
              self._unroll_split, port_B)
    )
    self._pull_data_thread.start()
    self._reply_model_thread = Thread(
        target=self._reply_model, args=(self._zmq_context, port_A))
    self._reply_model_thread.start()

  def run(self):
    self.coef_opp = get_schedule_fn(self.coef_opp_init, schedule=self.coef_opp_schedule)
    self.coef_adv = get_schedule_fn(self.coef_adv_init, schedule=self.coef_adv_schedule)
    self.coef_abs = get_schedule_fn(self.coef_abs_init, schedule=self.coef_abs_schedule)

    while len(self._episode_infos) < self._episode_infos.maxlen / 2:
      tprint('episode num is %d' %len(self._episode_infos))
      time.sleep(1)

    batch_queue = Queue(4)
    batch_threads = [
        Thread(target=self._prepare_batch,
               args=(self._data_queue, batch_queue,
                     self._batch_size * self._unroll_split))
        for _ in range(8)
    ]
    for thread in batch_threads:
      thread.start()

    updates, loss = 0, []
    time_start = time.time()
    while True:
      while (self._learn_act_speed_ratio > 0 and
          updates * self._batch_size >= \
          self._num_unrolls * self._learn_act_speed_ratio):
        time.sleep(0.001)
      updates += 1
      lr_now = self._lr(updates)
      # schedule the rate

      if self.coef_opp_schedule == 'const':
        coef_opp_now = self.coef_opp(0)
      elif self.coef_opp_schedule == 'linear':
        coef_opp_now = self.coef_opp(updates, 5 * 10e7)
      elif self.coef_opp_schedule == 'step':
        coef_opp_now = self.coef_opp(updates)

      if self.coef_adv_schedule == 'const':
        coef_adv_now = self.coef_adv(0)
      elif self.coef_adv_schedule == 'linear':
        coef_adv_now = self.coef_adv(updates, 5 * 10e7)
      elif self.coef_adv_schedule == 'step':
        coef_adv_now = self.coef_adv(updates)

      if self.coef_abs_schedule == 'const':
        coef_abs_now = self.coef_abs(0)
      elif self.coef_abs_schedule == 'linear':
        coef_abs_now = self.coef_abs(updates, 5 * 10e7)
      elif self.coef_adv_schedule == 'step':
        coef_abs_now = self.coef_abs(updates)

      clip_range_now = self._clip_range(updates)

      batch = batch_queue.get()

      obs, returns, dones, actions, values, neglogpacs, \
      opp_obs, opp_returns, opp_values, abs_returns, abs_values, \
      states, opp_states, abs_states = batch

      loss.append(self._model.train(lr_now, clip_range_now, coef_opp_now, coef_adv_now, coef_abs_now,
                                    obs, returns, dones, actions, values, neglogpacs, opp_obs,
                                    opp_returns, opp_values, abs_returns, abs_values,
                                    states, opp_states, abs_states))

      self._model_params = self._model.read_params()

      if updates % self._print_interval == 0:
        loss_mean = np.mean(loss, axis=0)
        batch_steps = self._batch_size * self._unroll_length
        time_elapsed = time.time() - time_start
        train_fps = self._print_interval * batch_steps / time_elapsed
        rollout_fps = len(self._data_timesteps) * self._unroll_length  / \
            (time.time() - self._data_timesteps[0])
        var = explained_variance(values, returns)
        avg_reward = safemean([info['r'] for info in self._episode_infos])
        avg_return = safemean(returns)
        # print the winning rate and number of the games
        total_game = len(self._episode_infos)
        win_game = sum([info['win'] for info in self._episode_infos])
        tie_game = sum([info['tie'] for info in self._episode_infos])
        loss_game = sum([info['loss'] for info in self._episode_infos])
        winning_rate = sum([info['win'] for info in self._episode_infos]) * 1.0 / total_game
        win_count_tie = (((win_game - loss_game) * 1.0 / (total_game)) + 1) / 2.0
        win_plus_tie = (win_game + tie_game) * 1.0 / (total_game)
        tprint('Total_Game is %d, Winning_rate is %f, Winning_rate_tie is %f, Winning_plus_tie is %f,'
               'win %d, tie %d, loss %d,'
               % (total_game, winning_rate, win_count_tie,  win_plus_tie, win_game, tie_game, loss_game))
        if self._save_dir is not None:
            os.makedirs(self._save_dir, exist_ok=True)
            fid = open(self._save_dir + '/Log.txt', 'a+')
            fid.write("%d %f %f %f %f %f\n" %(updates, winning_rate, win_count_tie, win_plus_tie, avg_reward, avg_return))
            fid.close()

        tprint("Update: %d	Train-fps: %.1f	Rollout-fps: %.1f	"
               "Explained-var: %.5f	Avg-reward %.2f	Policy-loss: %.5f	"
               "Value-loss: %.5f	Policy-entropy: %.5f	Approx-KL: %.5f	"
               "Clip-frac: %.3f	Time: %.1f" % (updates, train_fps, rollout_fps,
               var, avg_reward, *loss_mean[:5], time_elapsed))
        time_start, loss = time.time(), []

      if self._save_dir is not None and updates % self._save_interval == 0:
        os.makedirs(self._save_dir, exist_ok=True)
        save_path = os.path.join(self._save_dir, 'checkpoint-%d' % updates)
        self._model.save(save_path)
        tprint('Saved to %s.' % save_path)

  def _prepare_batch(self, data_queue, batch_queue, batch_size):
    while True:
      batch = random.sample(data_queue, batch_size)

      obs, opp_obs, returns, dones, actions, values, neglogpacs, \
      opp_returns, opp_values, abs_returns, abs_values, \
      states, opp_states, abs_states = zip(*batch)

      if isinstance(obs[0], tuple):
        obs = tuple(np.concatenate(ob) for ob in zip(*obs))
        opp_obs = tuple(np.concatenate(ob) for ob in zip(*opp_obs))
      else:
        obs = np.concatenate(obs)
        opp_obs = np.concatenate(opp_obs)

      returns = np.concatenate(returns)
      dones = np.concatenate(dones)
      actions = np.concatenate(actions)
      values = np.concatenate(values)
      neglogpacs = np.concatenate(neglogpacs)
      states = np.concatenate(states) if states[0] is not None else None
      opp_states = np.concatenate(opp_states) if opp_states[0] is not None else None
      abs_states = np.concatenate(abs_states) if abs_states[0] is not None else None

      opp_returns = np.concatenate(opp_returns)
      opp_values = np.concatenate(opp_values)
      abs_returns = np.concatenate(abs_returns)
      abs_values = np.concatenate(abs_values)

      # batch queue
      batch_queue.put((obs, returns, dones, actions, values, neglogpacs,
                       opp_obs, opp_returns, opp_values, abs_returns,
                       abs_values, states, opp_states, abs_states))

  def _pull_data(self, zmq_context, data_queue, episode_infos, unroll_split,
                 port_B):
    receiver = zmq_context.socket(zmq.PULL)
    receiver.setsockopt(zmq.RCVHWM, 1)
    receiver.setsockopt(zmq.SNDHWM, 1)
    receiver.bind("tcp://*:%s" % port_B)
    while True:
      data = receiver.recv_pyobj()
      if unroll_split > 1:
        # Added by Xian
        data_queue.extend(list(zip(*(
            [list(zip(*transform_tuple(
                data[0], lambda x: np.split(x, unroll_split))))] + \
            [list(zip(*transform_tuple(
                data[1], lambda x: np.split(x, unroll_split))))] + \
                [np.split(arr, unroll_split) for arr in data[2:-4]] + \
                [[data[-4] for _ in range(unroll_split)]] + \
                 [[data[-3] for _ in range(unroll_split)]] + \
                 [[data[-2] for _ in range(unroll_split)]]
        ))))
      else:
        data_queue.append(data[:-1])
      episode_infos.extend(data[-1])
      self._data_timesteps.append(time.time())
      self._num_unrolls += 1

  def _reply_model(self, zmq_context, port_A):
    receiver = zmq_context.socket(zmq.REP)
    receiver.bind("tcp://*:%s" % port_A)
    while True:
      msg = receiver.recv_string()
      assert msg == "request model"
      receiver.send_pyobj(self._model_params)


def safemean(xs):
  return np.nan if len(xs) == 0 else np.mean(xs)


def transform_tuple(x, transformer):
  if isinstance(x, tuple):
    return tuple(transformer(a) for a in x)
  else:
    return transformer(x)
# Add function


def get_schedule_fn(value_schedule, schedule):
  """
  Transform (if needed) learning rate and clip range
  to callable.
  :param value_schedule: (callable or float)
  :return: (function)
  """
  # If the passed schedule is a float
  # create a constant function
  if schedule == 'const':
    value_schedule = constfn(value_schedule)
  elif schedule == 'linear':
    value_schedule = linearfn(value_schedule)
  elif schedule == 'step':
    value_schedule = stepfn(value_schedule)
  else:
    assert callable(value_schedule)
  return value_schedule


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def swap_and_flatten(arr):
  """
  swap and then flatten axes 0 and 1
  :param arr: (np.ndarray)
  :return: (np.ndarray)
  """
  shape = arr.shape
  return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])


def linearfn(val):
  """
  :param val: (float)
  :return: (function)
  """

  def func(epoch, total_epoch):
    frac = 1.0 - (epoch - 1.0) / total_epoch
    return val * frac

  return func


def stepfn(val):
  """
  :param val: (float)
  :return: (function)
  """

  def func(epoch, drop=0.8, epoch_drop=400):
    ratio = drop ** ((epoch + 1) // epoch_drop)
    return val * ratio

  return func


def constfn(val):
  def f(_):
    return val
  return f



