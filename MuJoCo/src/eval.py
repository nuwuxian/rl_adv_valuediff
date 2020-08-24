from zoo_utils import LSTMPolicy, MlpPolicyValue
import gym
import gym_compete
import argparse
import tensorflow as tf
import numpy as np
from common import env_list
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.running_mean_std import RunningMeanStd
from zoo_utils import setFromFlat, load_from_file, load_from_model


def run(config):

    ENV_NAME = env_list[config.env]
    env = gym.make(ENV_NAME)

    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    sess = tf.Session(config=tf_config)
    sess.__enter__()

    policy = []

    if config.pi0_type == 'our':
        if config.pi0_nn_type == 'mlp':
            policy.append(MlpPolicyValue(scope="policy0", reuse=False,
                                         ob_space=env.observation_space.spaces[0],
                                         ac_space=env.action_space.spaces[0],
                                         hiddens=[64, 64], normalize=True))
        else:
            policy.append(LSTMPolicy(scope="policy0", reuse=False,
                                     ob_space=env.observation_space.spaces[0],
                                     ac_space=env.action_space.spaces[0],
                                     hiddens=[128, 128], normalize=True))
    else:
        policy.append(MlpPolicy(sess, env.observation_space.spaces[0], env.action_space.spaces[0], 1, 1, 1, reuse=False))

    if config.pi1_type == 'our':
        if config.pi1_nn_type == 'mlp':
            policy.append(MlpPolicyValue(scope="policy1", reuse=False,
                                         ob_space=env.observation_space.spaces[1],
                                         ac_space=env.action_space.spaces[1],
                                         hiddens=[64, 64], normalize=True))
        else:
            policy.append(LSTMPolicy(scope="policy1", reuse=False,
                                     ob_space=env.observation_space.spaces[1],
                                     ac_space=env.action_space.spaces[1],
                                     hiddens=[128, 128], normalize=True))
    else:
        policy.append(MlpPolicy(sess, env.observation_space.spaces[1], env.action_space.spaces[1], 1, 1, 1, reuse=False))

    sess.run(tf.variables_initializer(tf.global_variables()))

    # load running mean/variance and model for opp_agent (policy0)
    if config.pi0_type == 'our':
        if config.pi0_nn_type == 'mlp':
            # load running mean/variance and model

            none_trainable_list = policy[0].get_variables()[:6]
            shapes = list(map(lambda x: x.get_shape().as_list(), none_trainable_list))
            none_trainable_size = np.sum([int(np.prod(shape)) for shape in shapes])
            none_trainable_param = load_from_file(config.pi0_norm_path)[:none_trainable_size]
            if 'multiagent-competition' in config.pi0_path:
                trainable_param = load_from_file(config.pi0_path)[none_trainable_size:]
            else:
                trainable_param = load_from_model(param_pkl_path=config.pi0_path)
            param = np.concatenate([none_trainable_param, trainable_param], axis=0)
            setFromFlat(policy[0].get_variables(), param)
        else:
            none_trainable_list = policy[0].get_variables()[:12]
            shapes = list(map(lambda x: x.get_shape().as_list(), none_trainable_list))
            none_trainable_size = np.sum([int(np.prod(shape)) for shape in shapes])
            none_trainable_param = load_from_file(config.pi0_norm_path)[:none_trainable_size]
            if 'multiagent-competition' in config.pi0_path:
                trainable_param = load_from_file(config.pi0_path)[none_trainable_size:]
            else:
                trainable_param = load_from_model(param_pkl_path=config.pi0_path)
            param = np.concatenate([none_trainable_param, trainable_param], axis=0)
            setFromFlat(policy[0].get_variables(), param)

    else:
        pi0_obs_rms = load_from_file(config.pi0_norm_path)
        param = load_from_model(param_pkl_path=config.pi0_path)
        setFromFlat(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model'), param)

    if config.pi1_type == 'our':
        if config.pi1_nn_type == 'mlp':
            # load running mean/variance and model
            none_trainable_list = policy[1].get_variables()[:6]
            shapes = list(map(lambda x: x.get_shape().as_list(), none_trainable_list))
            none_trainable_size = np.sum([int(np.prod(shape)) for shape in shapes])
            none_trainable_param = load_from_file(config.pi1_norm_path)[:none_trainable_size]
            if 'multiagent-competition' in config.pi1_path:
                trainable_param = load_from_file(config.pi1_path)[none_trainable_size:]
            else:
                trainable_param = load_from_model(param_pkl_path=config.pi1_path)
            param = np.concatenate([none_trainable_param, trainable_param], axis=0)
            setFromFlat(policy[1].get_variables(), param)
        else:
            none_trainable_list = policy[1].get_variables()[:12]
            shapes = list(map(lambda x: x.get_shape().as_list(), none_trainable_list))
            none_trainable_size = np.sum([int(np.prod(shape)) for shape in shapes])
            none_trainable_param = load_from_file(config.pi1_norm_path)[:none_trainable_size]
            if 'multiagent-competition' in config.pi1_path:
                trainable_param = load_from_file(config.pi1_path)[none_trainable_size:]
            else:
                trainable_param = load_from_model(param_pkl_path=config.pi1_path)
            param = np.concatenate([none_trainable_param, trainable_param], axis=0)
            setFromFlat(policy[1].get_variables(), param)
    else:
        pi1_obs_rms = load_from_file(config.pi1_norm_path)
        param = load_from_model(param_pkl_path=config.pi1_path)
        setFromFlat(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model'), param)


    max_episodes = config.max_episodes
    num_episodes = 0
    nstep = 0
    total_reward = [0.0  for _ in range(len(policy))]
    total_scores = [0 for _ in range(len(policy))]

    # total_scores = np.asarray(total_scores)
    observation = env.reset()
    print("-"*5 + " Episode %d " % (num_episodes+1) + "-"*5)

    while num_episodes < max_episodes:
        # normalize the observation-0 and observation-1
        obs_0, obs_1 = observation
        if config.pi0_type == 'our':
            action_0 = policy[0].act(stochastic=True, observation=obs_0)[0]
        else:
            obs_0 = np.clip(
                (obs_0 - pi0_obs_rms.mean) / np.sqrt(pi0_obs_rms.var + 1e-8), -10, 10)
            action_0 = policy[0].step(obs=obs_0[None,:], deterministic=False)[0][0]

        if config.pi1_type == 'our':
            action_1 = policy[1].act(stochastic=True, observation=obs_1)[0]
        else:
            obs_1 = np.clip(
                (obs_1 - pi1_obs_rms.mean) / np.sqrt(pi1_obs_rms.var + 1e-8), -10, 10)
            action_1 = policy[1].step(obs=obs_1[None,:], deterministic=False)[0][0]

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
            if config.pi0_type == 'our' and config.pi0_nn_type == 'lstm':
                policy[0].reset() # this is specific for LSTM policy
            if config.pi1_type == 'our' and config.pi1_nn_type == 'lstm':
                policy[1].reset() # this is specific for LSTM policy

            if num_episodes < max_episodes:
                print("-"*5 + "Episode %d" % (num_episodes+1) + "-"*5)

    print('================')
    print("Game win_0 %.2f" %float(total_scores[0]/max_episodes))
    print("Game win_1 %.2f" %float(total_scores[1]/max_episodes))
    print("Game tie %.2f" %float((max_episodes - total_scores[0] - total_scores[1])/max_episodes))
    print('================')

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Environments for Multi-agent competition")
    p.add_argument("--max-episodes", default=100, help="max number of matches", type=int)
    p.add_argument("--epsilon", default=1e-8, type=float)
    p.add_argument("--clip_obs", default=10, type=float)

    ## Kick And Defend
    # p.add_argument("--env", default=3, type=int) # 2: YouShallNotPass # 3: KickAndDefend # 4: SumoAnts # 5: SumoHumans

    # victim-agent

    # zoo-victim
    # p.add_argument("--pi0_type", default="our", type=str)
    # p.add_argument("--pi0_nn_type", default="lstm", type=str)
    #
    # p.add_argument("--pi0_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/kick-and-defend/kicker/agent1_parameters-v1.pkl', type=str)
    # p.add_argument("--pi0_norm_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/kick-and-defend/kicker/agent1_parameters-v1.pkl", type=str)

    # our retrained vic.
    # p.add_argument("--pi0_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/our_attack/high_capacity/KickAndDefend/20200818_124108-0/KickAndDefend-v0.npy', type=str)
    # p.add_argument("--pi0_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/our_attack/high_capacity/KickAndDefend/20200818_124119-1/KickAndDefend-v0.npy', type=str)
    # p.add_argument("--pi0_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/our_attack/high_capacity/KickAndDefend/20200818_124130-2/KickAndDefend-v0.npy', type=str)
    # p.add_argument("--pi0_norm_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/kick-and-defend/kicker/agent1_parameters-v1.pkl", type=str)

    # ucb retrained vic.
    # p.add_argument("--pi0_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/ucb/KickAndDefend/20200819_150752-0/KickAndDefend-v0.npy', type=str)
    # p.add_argument("--pi0_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/ucb/KickAndDefend/20200819_150802-1/KickAndDefend-v0.npy', type=str)
    # p.add_argument("--pi0_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/ucb/KickAndDefend/20200819_150812-2/KickAndDefend-v0.npy', type=str)
    # p.add_argument("--pi0_norm_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/kick-and-defend/kicker/agent1_parameters-v1.pkl", type=str)

    # Adv agent
    # p.add_argument("--pi1_type", default="stable", type=str)
    # p.add_argument("--pi1_nn_type", default="mlp", type=str)

    # ucb adv.
    # p.add_argument("--pi1_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/adv_agent-zoo/ucb/kick/model.npy", type=str)
    # p.add_argument("--pi1_norm_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/adv_agent-zoo/ucb/kick/obs_rms.pkl", type=str)

    # our adv.
    # p.add_argument("--pi1_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/adv_agent-zoo/our_attack/kick/model.pkl", type=str)
    # p.add_argument("--pi1_norm_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/adv_agent-zoo/our_attack/kick/obs_rms.pkl", type=str)

    # zoo-agent
    # p.add_argument("--pi1_type", default="our", type=str)
    # p.add_argument("--pi1_nn_type", default="lstm", type=str)
    #
    # p.add_argument("--pi1_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/kick-and-defend/defender/agent2_parameters-v2.pkl", type=str)
    # p.add_argument("--pi1_norm_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/kick-and-defend/defender/agent2_parameters-v2.pkl", type=str)




    ## You Shall Not Pass
    # p.add_argument("--env", default=2, type=int) # 2: YouShallNotPass # 3: KickAndDefend # 4: SumoAnts # 5: SumoHumans
    #
    # Adv agent
    # p.add_argument("--pi0_type", default="stable", type=str)
    # p.add_argument("--pi0_nn_type", default="mlp", type=str)
    #
    # ucb adv.
    # p.add_argument("--pi0_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/adv_agent-zoo/ucb/you/model.npy", type=str)
    # p.add_argument("--pi0_norm_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/adv_agent-zoo/ucb/you/obs_rms.pkl", type=str)
    #
    # our adv.
    # p.add_argument("--pi0_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/adv_agent-zoo/our_attack/you/model.pkl", type=str)
    # p.add_argument("--pi0_norm_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/adv_agent-zoo/our_attack/you/obs_rms.pkl", type=str)
    #
    # zoo-agent
    # p.add_argument("--pi0_type", default="our", type=str)
    # p.add_argument("--pi0_nn_type", default="mlp", type=str)
    #
    # p.add_argument("--pi0_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/you-shall-not-pass/agent1_parameters-v1.pkl', type=str)
    # p.add_argument("--pi0_norm_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/you-shall-not-pass/agent1_parameters-v1.pkl", type=str) # YouShallNotPass agent2 is the runner
    #
    #
    # victim-agent
    #
    # zoo-victim
    # p.add_argument("--pi1_type", default="our", type=str)
    # p.add_argument("--pi1_nn_type", default="mlp", type=str)
    #
    # p.add_argument("--pi1_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/you-shall-not-pass/agent2_parameters-v1.pkl', type=str)
    # p.add_argument("--pi1_norm_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/you-shall-not-pass/agent2_parameters-v1.pkl", type=str) # YouShallNotPass agent2 is the runner
    #
    # our retrained vic.
    # p.add_argument("--pi1_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/our_attack/high_capacity/YouShallNotPass/20200816_125658-0/YouShallNotPassHumans-v0.pkl', type=str)
    # p.add_argument("--pi1_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/our_attack/high_capacity/YouShallNotPass/20200816_125708-1/YouShallNotPassHumans-v0.pkl', type=str)
    # p.add_argument("--pi1_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/our_attack/high_capacity/YouShallNotPass/20200816_125718-2/YouShallNotPassHumans-v0.pkl', type=str)
    # p.add_argument("--pi1_norm_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/you-shall-not-pass/agent2_parameters-v1.pkl", type=str) # YouShallNotPass agent2 is the runner
    #
    # ucb retrained vic.
    # p.add_argument("--pi1_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/ucb/YouShallNotPass/20200820_160129-0/YouShallNotPassHumans-v0.pkl', type=str)
    # p.add_argument("--pi1_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/ucb/YouShallNotPass/20200820_160139-1/YouShallNotPassHumans-v0.pkl', type=str)
    # p.add_argument("--pi1_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/ucb/YouShallNotPass/20200820_160151-2/YouShallNotPassHumans-v0.pkl', type=str)
    # p.add_argument("--pi1_norm_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/you-shall-not-pass/agent2_parameters-v1.pkl", type=str) # YouShallNotPass agent2 is the runner
    #



    # ## SumoHumans
    #
    p.add_argument("--env", default=5, type=int) # 2: YouShallNotPass # 3: KickAndDefend # 4: SumoAnts # 5: SumoHumans

    # victim-agent

    # zoo-victim
    p.add_argument("--pi0_type", default="our", type=str)
    p.add_argument("--pi0_nn_type", default="lstm", type=str)

    # p.add_argument("--pi0_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/sumo/humans/agent_parameters-v3.pkl', type=str)
    # p.add_argument("--pi0_norm_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/sumo/humans/agent_parameters-v3.pkl", type=str)

    # our retrained vic.
    # p.add_argument("--pi0_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/our_attack/high_capacity/SumoHumans/20200817_093255-0/SumoHumans-v0.npy', type=str)
    # p.add_argument("--pi0_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/our_attack/high_capacity/SumoHumans/20200817_093302-1/SumoHumans-v0.npy', type=str)
    # p.add_argument("--pi0_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/our_attack/high_capacity/SumoHumans/20200817_093313-2/SumoHumans-v0.npy', type=str)
    # p.add_argument("--pi0_norm_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/sumo/humans/agent_parameters-v3.pkl", type=str)

    # ucb retrained vic.
    p.add_argument("--pi0_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/ucb/SumoHumans/20200820_175047-0/SumoHumans-v0.pkl', type=str)
    # p.add_argument("--pi0_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/ucb/SumoHumans/20200820_175057-1/SumoHumans-v0.pkl', type=str)
    # p.add_argument("--pi0_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/ucb/SumoHumans/20200820_175107-2/SumoHumans-v0.pkl', type=str)
    p.add_argument("--pi0_norm_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/sumo/humans/agent_parameters-v3.pkl", type=str)

    # zoo-agent
    # p.add_argument("--pi1_type", default="our", type=str)
    # p.add_argument("--pi1_nn_type", default="lstm", type=str)
    #
    # p.add_argument("--pi1_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/sumo/humans/agent_parameters-v1.pkl', type=str)
    # p.add_argument("--pi1_norm_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/sumo/humans/agent_parameters-v1.pkl", type=str)

    # Adv agent
    p.add_argument("--pi1_type", default="stable", type=str)
    p.add_argument("--pi1_nn_type", default="mlp", type=str)

    # ucb adv.
    p.add_argument("--pi1_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/adv_agent-zoo/ucb/humans/model.pkl", type=str)
    p.add_argument("--pi1_norm_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/adv_agent-zoo/ucb/humans/obs_rms.pkl", type=str)

    # our adv.
    # p.add_argument("--pi1_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/adv_agent-zoo/our_attack/humans/model.pkl", type=str)
    # p.add_argument("--pi1_norm_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/adv_agent-zoo/our_attack/humans/obs_rms.pkl", type=str)




    ## SumoAnts

    # p.add_argument("--env", default=4, type=int) # 2: YouShallNotPass # 3: KickAndDefend # 4: SumoAnts # 5: SumoHumans

    # victim-agent

    # zoo-victim
    # p.add_argument("--pi0_type", default="our", type=str)
    # p.add_argument("--pi0_nn_type", default="lstm", type=str)

    # p.add_argument("--pi0_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/sumo/ants/agent_parameters-v1.pkl', type=str)
    # p.add_argument("--pi0_norm_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/sumo/ants/agent_parameters-v1.pkl', type=str)

    # our retrained vic.
    # p.add_argument("--pi0_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/our_attack/high_capacity/SumoAnts/20200818_124818-0/SumoAnts-v0.npy', type=str)
    # p.add_argument("--pi0_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/our_attack/high_capacity/SumoAnts/20200818_124825-1/SumoAnts-v0.npy', type=str)
    # p.add_argument("--pi0_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/our_attack/high_capacity/SumoAnts/20200818_124836-2/SumoAnts-v0.npy', type=str)
    # p.add_argument("--pi0_norm_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/sumo/ants/agent_parameters-v1.pkl', type=str)

    # ucb retrained vic.
    # p.add_argument("--pi0_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/ucb/SumoAnts/20200818_233706-0/SumoAnts-v0.npy', type=str)
    # p.add_argument("--pi0_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/ucb/SumoAnts/20200818_233716-1/SumoAnts-v0.npy', type=str)
    # p.add_argument("--pi0_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/results/retrain_50/ucb/SumoAnts/20200818_233727-2/SumoAnts-v0.npy', type=str)
    # p.add_argument("--pi0_norm_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/sumo/ants/agent_parameters-v1.pkl', type=str)

    # Adv agent
    # p.add_argument("--pi1_type", default="stable", type=str)
    # p.add_argument("--pi1_nn_type", default="mlp", type=str)

    # ucb adv.
    # p.add_argument("--pi1_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/adv_agent-zoo/ucb/ants/model.npy", type=str)
    # p.add_argument("--pi1_norm_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/adv_agent-zoo/ucb/ants/obs_rms.pkl", type=str)

    # our adv.
    # p.add_argument("--pi1_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/adv_agent-zoo/our_attack/ants/model.pkl", type=str)
    # p.add_argument("--pi1_norm_path", default="/Users/Henryguo/Desktop/rl_newloss/MuJoCo/adv_agent-zoo/our_attack/ants/obs_rms.pkl", type=str)

    # zoo-agent
    # p.add_argument("--pi1_type", default="our", type=str)
    # p.add_argument("--pi1_nn_type", default="lstm", type=str)
    #
    # p.add_argument("--pi1_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/sumo/ants/agent_parameters-v2.pkl', type=str)
    # p.add_argument("--pi1_norm_path", default='/Users/Henryguo/Desktop/rl_newloss/MuJoCo/multiagent-competition/agent-zoo/sumo/ants/agent_parameters-v2.pkl', type=str)




    config = p.parse_args()
    run(config)


