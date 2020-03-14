from zoo_utils import LSTMPolicy, MlpPolicyValue
import gym
import gym_compete
import pickle
import sys
import argparse
import tensorflow as tf
import numpy as np
from common import env_list

from stable_baselines.common.running_mean_std import RunningMeanStd
from zoo_utils import setFromFlat, load_from_file


def run(config):

    ENV_NAME = env_list[config.env]

    if ENV_NAME in ['multicomp/YouShallNotPassHumans-v0', "multicomp/RunToGoalAnts-v0", "multicomp/RunToGoalHumans-v0"]:
        policy_type="mlp"
    else:
        policy_type="lstm"

    env = gym.make(ENV_NAME)

    epsilon = config.epsilon
    clip_obs = config.clip_obs

    param_paths = [config.opp_path, config.vic_path]

    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    sess = tf.Session(config=tf_config)
    sess.__enter__()

    policy = []
    for i in range(2):
        scope = "policy" + str(i)
        if policy_type == "lstm":
            policy.append(LSTMPolicy(scope=scope, reuse=False,
                                     ob_space=env.observation_space.spaces[i],
                                     ac_space=env.action_space.spaces[i],
                                     hiddens=[128, 128], normalize=False))
        else:
            policy.append(MlpPolicyValue(scope=scope, reuse=False,
                                         ob_space=env.observation_space.spaces[i],
                                         ac_space=env.action_space.spaces[i],
                                         hiddens=[64, 64], normalize=True))

    # initialize uninitialized variables
    sess.run(tf.variables_initializer(tf.global_variables()))

    params = [load_from_file(param_pkl_path=path) for path in param_paths]
    for i in range(len(policy)):
        setFromFlat(policy[i].get_variables(), params[i])

    max_episodes = config.max_episodes
    num_episodes = 0
    nstep = 0
    total_reward = [0.0  for _ in range(len(policy))]
    total_scores = [0 for _ in range(len(policy))]


    obs0_rms = RunningMeanStd(shape=env.observation_space.spaces[0].shape)
    obs1_rms = RunningMeanStd(shape=env.observation_space.spaces[1].shape)

    # total_scores = np.asarray(total_scores)
    observation = env.reset()
    print("-"*5 + " Episode %d " % (num_episodes+1) + "-"*5)
    while num_episodes < max_episodes:

    #   env.render()

        # normalize the observation-0 and observation-1

        obs0_rms.update(observation[0].copy())
        obs_0 = np.clip((observation[0] - obs0_rms.mean) / np.sqrt(obs0_rms.var + epsilon),
                                 -clip_obs, clip_obs)

        obs1_rms.update(observation[1].copy())
        obs_1 = np.clip((observation[1] - obs1_rms.mean) / np.sqrt(obs0_rms.var + epsilon),
                                 -clip_obs, clip_obs)

        action_0 = policy[0].act(stochastic=True, observation=obs_0)[0]
        action_1 = policy[1].act(stochastic=True, observation=obs_1)[0]
        action = (action_0, action_1)

        observation, reward, done, infos = env.step(action)
        nstep += 1
        for i in range(len(policy)):
            total_reward[i] += reward[i]
        if done[0]:
            num_episodes += 1
            draw = True
            for i in range(len(policy)):
                if 'winner' in infos[i]:
                    draw = False
                    total_scores[i] += 1
                    print("Winner: Agent {}, Scores: {}, Total Episodes: {}".format(i, total_scores, num_episodes))
            if draw:
                print("Game Tied: Agent {}, Scores: {}, Total Episodes: {}".format(i, total_scores, num_episodes))
            observation = env.reset()
            nstep = 0
            total_reward = [0.0  for _ in range(len(policy))]
            for i in range(len(policy)):
                policy[i].reset()
            if num_episodes < max_episodes:
                print("-"*5 + "Episode %d" % (num_episodes+1) + "-"*5)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Environments for Multi-agent competition")
    p.add_argument("--env", default=2, type=int)
    p.add_argument("--opp-path", default="../multiagent-competition/agent-zoo/you-shall-not-pass/agent1_parameters-v1pkl", type=str)
    p.add_argument("--vic_path", default="../multiagent-competition/agent-zoo/you-shall-not-pass/agent2_parameters-v1.pkl", type=str)

    p.add_argument("--max-episodes", default=500, help="max number of matches", type=int)
    p.add_argument("--epsilon", default=1e-8, type=float)
    p.add_argument("--clip_obs", default=10, type=float)


    config = p.parse_args()
    run(config)