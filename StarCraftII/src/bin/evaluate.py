from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import random

from absl import app
from absl import flags
from absl import logging

from envs.raw_env import SC2RawEnv
from envs.actions.zerg_action_wrappers import ZergActionWrapper
from envs.observations.zerg_observation_wrappers import ZergObservationWrapper
from utils.utils import print_arguments, print_actions, print_action_distribution
from agents.random_agent import RandomAgent
from agents.keyboard_agent import KeyboardAgent


FLAGS = flags.FLAGS
# total time steps.
flags.DEFINE_integer("num_episodes", 10, "Number of episodes to evaluate.")
flags.DEFINE_enum("agent", 'ppo', ['ppo', 'dqn', 'random', 'keyboard'],
                  "Agent name.")
flags.DEFINE_enum("policy", 'mlp', ['mlp', 'lstm'], "Job type.")
flags.DEFINE_string("game_version", '4.6', "Game core version.")
flags.DEFINE_integer("step_mul", 32, "Game steps per agent step.")
flags.DEFINE_enum("difficulty", '1',
                  ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A'],
                  "Bot's strength.")
flags.DEFINE_string("model_path", None, "Filepath to load initial model.")
flags.DEFINE_boolean("disable_fog", False, "Disable fog-of-war.")
flags.DEFINE_boolean("use_all_combat_actions", False, "Use all combat actions.")
flags.DEFINE_boolean("use_region_features", False, "Use region features")
flags.DEFINE_boolean("use_action_mask", True, "Use action mask or not.")
flags.FLAGS(sys.argv)


def create_env(random_seed=None):
    #  create environment.
    env = SC2RawEnv(map_name='AbyssalReef',
                  step_mul=FLAGS.step_mul,
                  agent_race='zerg',
                  bot_race='zerg',
                  difficulty=FLAGS.difficulty,
                  disable_fog=FLAGS.disable_fog,
                  random_seed=random_seed)
    # wrap agent action.
    env = ZergActionWrapper(env,
                          game_version=FLAGS.game_version,
                          mask=FLAGS.use_action_mask,
                          use_all_combat_actions=FLAGS.use_all_combat_actions)
    # wrap observation.
    env = ZergObservationWrapper(
      env,
      use_spatial_features=False,
      use_game_progress=(not FLAGS.policy == 'lstm'),
      action_seq_len=1 if FLAGS.policy == 'lstm' else 8,
      use_regions=FLAGS.use_region_features)
    print_actions(env)
    return env


def create_ppo_agent(env):
    import tensorflow as tf
    import multiprocessing
    from agents.ppo_policies import LstmPolicy, MlpPolicy
    from agents.ppo_agent import PPOAgent

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                      intra_op_parallelism_threads=ncpu,
                      inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()
    # define policy network type.
    policy = {'lstm': LstmPolicy, 'mlp': MlpPolicy}[FLAGS.policy]
    # define the ppo agent.
    agent = PPOAgent(env=env, policy=policy, model_path=FLAGS.model_path)
    return agent


def evaluate():
    game_seed =  random.randint(0, 2**32 - 1)
    print("Game Seed: %d" % game_seed)
    env = create_env(game_seed)

    if FLAGS.agent == 'ppo':
        agent = create_ppo_agent(env)
    elif FLAGS.agent == 'random':
        agent = RandomAgent(action_space=env.action_space)
    elif FLAGS.agent == 'keyboard':
        agent = KeyboardAgent(action_space=env.action_space)
    else:
        raise NotImplementedError

    try:
        cum_return = 0.0
        action_counts = [0] * env.action_space.n # number of possible actions.
        for i in range(FLAGS.num_episodes):
              observation = env.reset()
              agent.reset()
              done, step_id = False, 0
              while not done:
                    action = agent.act(observation)
                    print("Step ID: %d	Take Action: %d" % (step_id, action))
                    observation, reward, done, _ = env.step(action)
                    action_counts[action] += 1
                    cum_return += reward
                    step_id += 1
              print_action_distribution(env, action_counts)
              print("Evaluated %d/%d Episodes Avg Return %f Avg Winning Rate %f" % (
                  i + 1, FLAGS.num_episodes, cum_return / (i + 1),
                  ((cum_return / (i + 1)) + 1) / 2.0))
    except KeyboardInterrupt: pass
    finally: env.close()


def main(argv):
    logging.set_verbosity(logging.ERROR)
    print_arguments(FLAGS)
    evaluate()


if __name__ == '__main__':
    app.run(main)
