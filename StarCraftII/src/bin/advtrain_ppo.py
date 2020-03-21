from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')
from threading import Thread
import os
import multiprocessing
import random
import time

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from os import path as osp
import datetime

from agents.ppo_policies import LstmPolicy, MlpPolicy
from agents.ppo_values import LstmValue, MlpValue
from agents.ppo2_wrap import PPO_AdvActor, Adv_Learner

from envs.raw_env import SC2RawEnv
from envs.selfplay_raw_env import SC2SelfplayRawEnv
from envs.actions.zerg_action_wrappers import ZergActionWrapper, ZergPlayerActionWrapper
from envs.observations.zerg_observation_wrappers import ZergObservationWrapper, ZergPlayerObservationWrapper
from envs.rewards.reward_wrappers import RewardShapingWrapperV1, RewardShapingWrapperV2, KillingRewardWrapper
from utils.utils import print_arguments


FLAGS = flags.FLAGS

# game environment related hyperparameters.
flags.DEFINE_enum("job_name", 'learner', ['actor', 'learner'], "Job type.")
flags.DEFINE_string("learner_ip", "localhost", "Learner IP address.")
flags.DEFINE_string("port_A", "5700", "Port for transporting model.")
flags.DEFINE_string("port_B", "5701", "Port for transporting data.")
flags.DEFINE_float("learn_act_speed_ratio", 0, "Maximum learner/actor ratio.")
flags.DEFINE_string("game_version", '4.6', "Game core version.")
flags.DEFINE_integer("step_mul", 32, "Game steps per agent step.")
flags.DEFINE_string("difficulties", '1,2,4,6,9,A', "Bot's strengths.")
flags.DEFINE_boolean("disable_fog", True, "Disable fog-of-war.")
flags.DEFINE_boolean("use_all_combat_actions", False, "Use all combat actions.")
flags.DEFINE_boolean("use_region_features", False, "Use region features")
flags.DEFINE_boolean("use_action_mask", True, "Use region-wise combat.")
# reward shaping
flags.DEFINE_string("reward_shaping_type", "None", "type of reward shaping.")

# opponent model related hyperparameters.
flags.DEFINE_string("opp_model_path", '/home/wenbo/target-agent/checkpoint-1050000-2', "Opponent Model Path")
flags.DEFINE_boolean("use_victim_ob", False, "whether use victim obs")

# loss function related hyperparameters
flags.DEFINE_float("discount_gamma", 0.998, "Discount factor.")
flags.DEFINE_float("lambda_return", 0.95, "Lambda return factor.")
flags.DEFINE_float("clip_range", 0.1, "Clip range for PPO.")
flags.DEFINE_float("ent_coef", 0.01, "Coefficient for the entropy term.")
flags.DEFINE_float("vf_coef", 0.5, "Coefficient for the value loss.")

flags.DEFINE_integer("vic_coef_init", 1, "vic_coef_values")
flags.DEFINE_string("vic_coef_sch", 'const', "vic_coef_function")
flags.DEFINE_integer("adv_coef_init", -1, "adv_coef_values")
flags.DEFINE_string("adv_coef_sch", 'const', "adv_coef_function")
flags.DEFINE_integer("diff_coef_init", 0, "diff_coef_values")
flags.DEFINE_string("diff_coef_sch", 'const', "diff_coef_sch")

# adversarial agent
flags.DEFINE_enum("policy", 'mlp', ['mlp', 'lstm'], "Job type.")
flags.DEFINE_enum("value", 'mlp', ['mlp', 'lstm'], "Value type")

# learning process.
flags.DEFINE_integer("unroll_length", 128, "Length of rollout steps.") # training batch size for mlp.
flags.DEFINE_integer("learner_queue_size", 1024, "Size of learner's unroll queue per update.")
flags.DEFINE_integer("max_episode", 400, "num of games per update.")
flags.DEFINE_integer("game_steps_per_episode", 43200, "Maximum steps per episode.")
flags.DEFINE_integer("batch_size", 8, "Batch size.") # batch_size * unroll_length
flags.DEFINE_float("learning_rate", 1e-5, "Learning rate.")

# save and print.
flags.DEFINE_string("init_model_path", '/home/wenbo/target-agent/checkpoint-100000', "Initial model path.")
flags.DEFINE_string("save_dir", "/home/wenbo/adv_shape_v1/", "Dir to save models to")
flags.DEFINE_integer("save_interval", 50000, "Model saving frequency.")
flags.DEFINE_integer("print_interval", 1000, "Print train cost frequency.")
flags.FLAGS(sys.argv)

random.seed(1234)
GAME_SEED = random.randint(0, 2 ** 32 - 1)


def make_timestamp():
    ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
    return datetime.datetime.now().strftime(ISO_TIMESTAMP)


def tf_config(ncpu=None):
    if ncpu is None:
        ncpu = multiprocessing.cpu_count()
        if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()


def create_env(difficulty, random_seed=None):
    env = SC2RawEnv(map_name='Flat64',
                    step_mul=FLAGS.step_mul,
                    resolution=16,
                    agent_race='zerg',
                    bot_race='zerg',
                    difficulty=difficulty,
                    disable_fog=FLAGS.disable_fog,
                    tie_to_lose=False,
                    game_steps_per_episode=FLAGS.game_steps_per_episode,
                    random_seed=random_seed)

    env = ZergActionWrapper(env,
                            game_version=FLAGS.game_version,
                            mask=FLAGS.use_action_mask,
                            use_all_combat_actions=FLAGS.use_all_combat_actions)
    env = ZergObservationWrapper(env,
                                 use_spatial_features=False,
                                 use_game_progress=(not FLAGS.policy == 'lstm'),
                                 action_seq_len=1 if FLAGS.policy == 'lstm' else 8,
                                 use_regions=FLAGS.use_region_features)
    print(env.observation_space, env.action_space)
    return env


def create_selfplay_env(random_seed=None):
    env = SC2SelfplayRawEnv(map_name='Flat64',
                            step_mul=FLAGS.step_mul,
                            resolution=16,
                            agent_race='zerg',
                            opponent_race='zerg',
                            tie_to_lose=False,
                            disable_fog=FLAGS.disable_fog,
                            game_steps_per_episode=FLAGS.game_steps_per_episode,
                            random_seed=random_seed)

    env = ZergPlayerActionWrapper(player=0,
                                  env=env,
                                  game_version=FLAGS.game_version,
                                  mask=FLAGS.use_action_mask,
                                  use_all_combat_actions=FLAGS.use_all_combat_actions)

    env = ZergPlayerObservationWrapper(player=0,
                                       env=env,
                                       use_spatial_features=False,
                                       use_game_progress=(not FLAGS.policy == 'lstm'),
                                       action_seq_len=1 if FLAGS.policy == 'lstm' else 8,
                                       use_regions=FLAGS.use_region_features)

    env = ZergPlayerActionWrapper(player=1,
                                  env=env,
                                  game_version=FLAGS.game_version,
                                  mask=FLAGS.use_action_mask,
                                  use_all_combat_actions=FLAGS.use_all_combat_actions)


    env = ZergPlayerObservationWrapper(player=1,
                                       env=env,
                                       use_spatial_features=False,
                                       use_game_progress=(not FLAGS.policy == 'lstm'),
                                       action_seq_len=1 if FLAGS.policy == 'lstm' else 8,
                                       use_regions=FLAGS.use_region_features)
    print(env.observation_space, env.action_space)
    print(env.observation_space, env.action_space)
    return env


def start_actor():
    tf_config(ncpu=2)
    env = create_selfplay_env(GAME_SEED)

    policy = {'lstm': LstmPolicy,
              'mlp': MlpPolicy}[FLAGS.policy]

    value = {'lstm': LstmValue,
             'mlp': MlpValue}[FLAGS.value]

    actor = PPO_AdvActor(env=env,
                         policy=policy,
                         value=value,
                         unroll_length=FLAGS.unroll_length,
                         gamma=FLAGS.discount_gamma,
                         lam=FLAGS.lambda_return,
                         learner_ip=FLAGS.learner_ip,
                         port_A=FLAGS.port_A,
                         port_B=FLAGS.port_B,
                         reward_shape=FLAGS.reward_shaping_type,
                         use_victim_ob=FLAGS.use_victim_ob,
                         victim_model=FLAGS.opp_model_path)

    actor.run()
    env.close()


def start_learner():
    tf_config()
    # does the difficulty matters ??
    env = create_env('1', GAME_SEED)
    policy = {'lstm': LstmPolicy,
            'mlp': MlpPolicy}[FLAGS.policy]

    value = {'lstm': LstmValue,
           'mlp': MlpValue}[FLAGS.value]

    # Change the dir_name
    FLAGS.save_dir = FLAGS.save_dir  + FLAGS.vic_coef_sch + "_" + str(FLAGS.vic_coef_init) + \
            "_" + FLAGS.adv_coef_sch + "_" + str(FLAGS.adv_coef_init) + "_" + FLAGS.diff_coef_sch + \
            "_" + str(FLAGS.diff_coef_init) + "_" + str(FLAGS.reward_shaping_type)
    timestamp = make_timestamp()

    FLAGS.save_dir = osp.join(FLAGS.save_dir, '{}-{}'.format(timestamp, str(GAME_SEED)))

    learner = Adv_Learner(env=env,
                          policy=policy,
                          value=value,
                          unroll_length=FLAGS.unroll_length,
                          lr=FLAGS.learning_rate,
                          clip_range=FLAGS.clip_range,
                          batch_size=FLAGS.batch_size,
                          ent_coef=FLAGS.ent_coef,
                          vf_coef=FLAGS.vf_coef,
                          max_grad_norm=0.5,
                          queue_size=FLAGS.learner_queue_size,
                          print_interval=FLAGS.print_interval,
                          save_interval=FLAGS.save_interval,
                          learn_act_speed_ratio=FLAGS.learn_act_speed_ratio,
                          save_dir=FLAGS.save_dir,
                          init_model_path=FLAGS.init_model_path,
                          port_A=FLAGS.port_A,
                          port_B=FLAGS.port_B,
                          max_episode=FLAGS.max_episode,
                          coef_opp_init=FLAGS.vic_coef_init,
                          coef_opp_schedule=FLAGS.vic_coef_sch,
                          coef_adv_init=FLAGS.adv_coef_init,
                          coef_adv_schedule=FLAGS.adv_coef_sch,
                          coef_abs_init=FLAGS.diff_coef_init,
                          coef_abs_schedule=FLAGS.diff_coef_sch,
                          )
    learner.run()
    env.close()


def main(argv):
    logging.set_verbosity(logging.ERROR)
    print_arguments(FLAGS)
    if FLAGS.job_name == 'actor': start_actor()
    elif FLAGS.job_name == 'learner': start_learner()
    else: print('Not support this flag.')


if __name__ == '__main__':
    app.run(main)
