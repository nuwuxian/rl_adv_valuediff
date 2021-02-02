from zoo_utils import LSTMPolicy, MlpPolicyValue
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
import gym
import gym_compete
import argparse
import tensorflow as tf
import numpy as np
from common import env_list
from zoo_utils import setFromFlat, load_from_file, load_from_model
import os

def run(config):
    ENV_NAME = env_list[config.env]
    if ENV_NAME in ['multicomp/YouShallNotPassHumans-v0']:
        policy_type="mlp"
        vic_id = 1
    else:
        policy_type="lstm"
        vic_id = 0
    env = gym.make(ENV_NAME)
    out_dir = config.out_dir + ENV_NAME.split('/')[1][:-3]
    os.makedirs(out_dir, exist_ok=True)

    opp_type = config.opp_type

    if vic_id == 1:
        param_paths = [config.opp_path, config.vic_path]
    else:
        param_paths = [config.vic_path, config.opp_path]

    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    sess = tf.Session(config=tf_config)
    sess.__enter__()

    agent_type = []
    policy = []
    for i in range(2):
        if i == vic_id:
            agent_type.append('norm')
        elif opp_type == 'norm':
            agent_type.append('norm')
        else:
            agent_type.append('adv')

    for i in range(2):
        if agent_type[i] == 'norm':
            if policy_type == 'mlp':
                policy.append(MlpPolicyValue(scope="policy" + str(i), reuse=False,
                                         ob_space=env.observation_space.spaces[i],
                                         ac_space=env.action_space.spaces[i],
                                         hiddens=[64, 64], normalize=True))
            else:
                policy.append(LSTMPolicy(scope="policy" + str(i), reuse=False,
                                         ob_space=env.observation_space.spaces[i],
                                         ac_space=env.action_space.spaces[i],
                                         hiddens=[128, 128], normalize=True))
        else:
            policy.append(MlpPolicy(sess, env.observation_space.spaces[i], env.action_space.spaces[i], 
                          1, 1, 1, reuse=False))


    # initialize uninitialized variables
    sess.run(tf.variables_initializer(tf.global_variables()))

    for i in range(2):

        if agent_type[i] == 'adv':    
            param = load_from_model(param_pkl_path=param_paths[i])
            adv_agent_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')
            setFromFlat(adv_agent_variables, param)
        
        else:
            param = load_from_file(param_pkl_path=param_paths[i])
            setFromFlat(policy[i].get_variables(), param)


    # max_steps and num_steps
    max_steps = config.max_steps
    num_steps = 0
    num_episodes = 0
    
    nstep = 0
    total_reward = [0.0  for _ in range(len(policy))]
    total_scores = [0 for _ in range(len(policy))]

    # norm path:
    obs_rms = load_from_file(config.norm_path)

    observation = env.reset()
    print("-"*5 + " Episode %d " % (num_episodes+1) + "-"*5)


    ret_act = []

    while num_steps < max_steps:
        actions = []

        for i, obs in enumerate(observation):
            if i == vic_id:
                # define policy activation
                act, _, ff_acts = policy[i].act(stochastic=True, observation=obs, extra_op=True)
                ret_act.append(ff_acts)
            elif agent_type[i] == 'adv':
                obs = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)
                act = policy[i].step(obs=obs[None, :], deterministic=False)[0][0]
            else:
                act = policy[i].act(stochastic=True, observation=obs)[0]

            actions.append(act)

        actions = tuple(actions)

        num_steps += 1
        observation, reward, done, infos = env.step(actions)
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

            for i in range(len(agent_type)):
                if agent_type[i] == 'norm':
                    policy[i].reset()

    env.close()
    # save ret_acts
    ret_act = np.vstack(ret_act)
    np.save(out_dir + '/activations_' + str(opp_type), ret_act)


if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Environments for Multi-agent competition")
    p.add_argument("--env", default=5, type=int)

    # YouShallNotPass
    # p.add_argument("--opp-path", default="../adv-agent/our_attack/you/model.pkl", type=str)
    # #p.add_argument("--opp_path", default="../multiagent-competition/agent-zoo/you-shall-not-pass/agent1_parameters-v1.pkl", type=str)
    # p.add_argument("--vic_path", default="../multiagent-competition/agent-zoo/you-shall-not-pass/agent2_parameters-v1.pkl", type=str)
    # p.add_argument("--norm_path", default="../adv-agent/our_attack/you/obs_rms.pkl", type=str)

    # KickAndDefend
    # p.add_argument("--opp-path", default="../adv-agent/our_attack/kick/model.pkl", type=str)
    # #p.add_argument("--opp_path", default="../multiagent-competition/agent-zoo/kick-and-defend/defender/agent2_parameters-v1.pkl", type=str)
    # p.add_argument("--vic_path", default="../multiagent-competition/agent-zoo/kick-and-defend/kicker/agent1_parameters-v1.pkl", type=str)
    # p.add_argument("--norm_path", default="../adv-agent/our_attack/kick/obs_rms.pkl", type=str)

    # SumoHumans
    p.add_argument("--opp-path", default="../adv-agent/our_attack/humans/model.pkl", type=str)
    #p.add_argument("--opp_path", default="../multiagent-competition/agent-zoo/sumo/humans/agent_parameters-v3.pkl", type=str)
    p.add_argument("--vic_path", default="../multiagent-competition/agent-zoo/sumo/humans/agent_parameters-v3.pkl", type=str)
    p.add_argument("--norm_path", default="../adv-agent/our_attack/humans/obs_rms.pkl", type=str)

    # SumoAnts
    # p.add_argument("--opp-path", default="../adv-agent/our_attack/ants/model.pkl", type=str)
    # #p.add_argument("--opp_path", default="../multiagent-competition/agent-zoo/sumo/ants/agent_parameters-v1.pkl", type=str)
    # p.add_argument("--vic_path", default="../multiagent-competition/agent-zoo/sumo/ants/agent_parameters-v1.pkl", type=str)
    # p.add_argument("--norm_path", default="../adv-agent/our_attack/ants/obs_rms.pkl", type=str)

    # either 'norm', 'ucb_adv', 'our_adv'
    p.add_argument('--opp_type', default='our_adv', type=str)
    p.add_argument("--max_steps", default=5000, help="max number of steps", type=int)
    p.add_argument("--epsilon", default=1e-8, type=float)
    p.add_argument("--clip_obs", default=10, type=float)
    p.add_argument("--out_dir", default='../activations/tsne/', type=str)

    config = p.parse_args()
    run(config)
