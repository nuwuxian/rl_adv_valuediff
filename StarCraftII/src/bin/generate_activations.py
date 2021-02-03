from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl import app, flags, logging

import numpy as np
import tensorflow as tf
import multiprocessing
from agents.ppo_policies import LstmPolicy, MlpPolicy
from agents.ppo_agent import PPOAgent

from envs.selfplay_raw_env import SC2SelfplayRawEnv
from envs.actions.zerg_action_wrappers import ZergPlayerActionWrapper
from envs.observations.zerg_observation_wrappers \
    import ZergPlayerObservationWrapper


from utils.utils import print_arguments, print_actions, print_action_distribution
from agents.random_agent import RandomAgent
from agents.keyboard_agent import KeyboardAgent


FLAGS = flags.FLAGS
# total time steps.
flags.DEFINE_integer("num_episodes", 50, "Number of episodes to evaluate.")
flags.DEFINE_enum("difficulty", '1',
                  ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A'],
                  "Bot's strength.")

flags.DEFINE_string("model_path", "/home/xkw5132/wuxian/rl_newloss/StarCraftII/adv-agent/our_attack/checkpoint-1050000-1", "Filepath to load initial model.")
flags.DEFINE_string("model_type", "our_adv")
flags.DEFINE_string("victim_path", "/home/xkw5132/wuxian/rl_newloss/StarCraftII/normal-agent/checkpoint-100000", "victim_path")
flag.DEFINE_string("out_path", '../../results/gmm')


flags.DEFINE_boolean("disable_fog", True, "Disable fog-of-war.")

flags.DEFINE_enum("agent", 'ppo', ['ppo', 'dqn', 'random', 'keyboard'],
                  "Agent name.")
flags.DEFINE_enum("policy", 'mlp', ['mlp', 'lstm'], "Job type.")
flags.DEFINE_enum("value", 'mlp', ['mlp', 'lstm'], "Value type")
flags.DEFINE_string("game_version", '4.6', "Game core version.")
flags.DEFINE_integer("step_mul", 32, "Game steps per agent step.")
flags.DEFINE_integer("max_timesteps", 40000, "Maximum steps to collect victim activations")
flags.DEFINE_integer("game_steps_per_episode", 43200, "Maximum steps per episode.")
flags.DEFINE_boolean("use_all_combat_actions", False, "Use all combat actions.")
flags.DEFINE_boolean("use_region_features", False, "Use region features")
flags.DEFINE_boolean("use_action_mask", True, "Use action mask or not.")
flags.FLAGS(sys.argv)

SAVE_PATH = FLAGS.out_path
GAME_SEED = 1234


def create_env(random_seed=None):
    env = SC2SelfplayRawEnv(map_name='Flat64',
                            step_mul=FLAGS.step_mul,
                            resolution=16,
                            agent_race='zerg',
                            opponent_race='zerg',
                            tie_to_lose=False,
                            disable_fog=FLAGS.disable_fog,
                            game_steps_per_episode=FLAGS.game_steps_per_episode,
                            random_seed=random_seed)

    env = ZergPlayerActionWrapper(
        player=0,
        env=env,
        game_version=FLAGS.game_version,
        mask=FLAGS.use_action_mask,
        use_all_combat_actions=FLAGS.use_all_combat_actions)
    env = ZergPlayerObservationWrapper(
        player=0,
        env=env,
        use_spatial_features=False,
        use_game_progress=(not FLAGS.policy == 'lstm'),
        action_seq_len=1 if FLAGS.policy == 'lstm' else 8,
        use_regions=FLAGS.use_region_features)

    env = ZergPlayerActionWrapper(
        player=1,
        env=env,
        game_version=FLAGS.game_version,
        mask=FLAGS.use_action_mask,
        use_all_combat_actions=FLAGS.use_all_combat_actions)
    env = ZergPlayerObservationWrapper(
        player=1,
        env=env,
        use_spatial_features=False,
        use_game_progress=(not FLAGS.policy == 'lstm'),
        action_seq_len=1 if FLAGS.policy == 'lstm' else 8,
        use_regions=FLAGS.use_region_features)
    print(env.observation_space, env.action_space)
    return env


def create_ppo_agent(env, model_path, scope_name):

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
    agent = PPOAgent(env=env, policy=policy, scope_name=scope_name, model_path=model_path)
    return agent


def evaluate(game_seed):
    env = create_env(game_seed)

    if FLAGS.agent == 'ppo':
        agent = create_ppo_agent(env, FLAGS.model_path, "model")
    elif FLAGS.agent == 'random':
        agent = RandomAgent(action_space=env.action_space)
    elif FLAGS.agent == 'keyboard':
        agent = KeyboardAgent(action_space=env.action_space)
    else:
        raise NotImplementedError

    try:
        win_games = 0
        loss_games = 0
        tie_games = 0
        victim_agent = create_ppo_agent(env, FLAGS.victim_path, "opponent_model")
        cum_return = 0.0
        action_counts = [0] * env.action_space.n # number of possible actions.

        total_steps = 0

        ff_acts = []
        while total_steps <= FLAGS.max_timesteps:
              observation_0, observation_1 = env.reset()
              agent.reset()
              victim_agent.reset()
              done, step_id = False, 0
              while not done:
                    total_steps += 1
                    action_0, _ = agent.act(observation_0)
                    action_1, vic_ff_acts = victim_agent.act(observation_1)
                    if total_steps <= FLAGS.max_timesteps:
                       ff_acts.append(vic_ff_acts)
                    (observation_0, observation_1), reward, done, _ = env.step([action_0, action_1])
                    action_counts[action_0] += 1
                    cum_return += reward
                    step_id += 1
              if done:
                  if reward == 1:
                      win_games += 1
                  elif reward == -1:
                      loss_games += 1
                  else:
                      tie_games += 1

              print("Evaluated %d/%d Episodes Avg Return %f Avg Winning Rate %f Win %d Lose %d tie %d" % (i + 1, FLAGS.num_episodes, cum_return / (i + 1), ((cum_return / (i + 1)) + 1) / 2.0, win_games, loss_games, tie_games))

        # save the ff_acts
        ff_acts = np.vstack(ff_acts)
        os.makedirs(SAVE_PATH, exist_ok=True)
        np.save(SAVE_PATH + '/activations_' + str(FLAGS.model_type), ff_acts)
        return (((cum_return / (FLAGS.num_episodes)) + 1) / 2.0, cum_return, win_games, loss_games, tie_games)
    except KeyboardInterrupt: pass
    finally: env.close()


def main(argv):
    logging.set_verbosity(logging.ERROR)
    winning_info = evaluate(GAME_SEED)
    victim = FLAGS.victim_path.split('/')[-1]
    model = FLAGS.model_path.split('/')[-1]
if __name__ == '__main__':
    app.run(main)
