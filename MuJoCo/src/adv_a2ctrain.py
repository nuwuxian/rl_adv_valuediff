import os
import argparse
import gym
import os.path as osp
from common import env_list
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from scheduling import ConstantAnnealer, Scheduler
from shaping_wrappers import apply_reward_wrapper
from environment import make_zoo_multi2single_env, Monitor
from logger import setup_logger
from ppo2_a2c import PPO_A2C
from value import MlpValue, MlpLstmValue
# from common import get_zoo_path


##################
# Hyper-parameters
##################
parser = argparse.ArgumentParser()
# game env
parser.add_argument("--env", type=int, default=3)
# random seed
parser.add_argument("--seed", type=int, default=0)
# number of game environment. should be dividable by NBATCHES if using a LSTM policy
parser.add_argument("--n_games", type=int, default=2) # N_GAME = 8
# which victim agent to use
parser.add_argument("--vic_agt_id", type=int, default=1)

# 2: YouShallNotPass
# victim agent id: 1

# 3: KickAndDefend
# victim agent id: 1

# 4: SumoAnts
# victim agent id: 1

# 5: SumoHumans
# victim agent id: 3


# victim agent network
parser.add_argument("--vic_net", type=str, default='MLP')
# adv agent network
parser.add_argument("--adv_net", type=str, default='MLP')

# learning rate scheduler
parser.add_argument("--lr_sch", type=str, default='linear')
# number of steps / lstm length should be small
parser.add_argument("--nsteps", type=int, default=2048)

# victim loss coefficient.
parser.add_argument("--vic_coef_init", type=int, default=0) # positive
# victim loss schedule
parser.add_argument("--vic_coef_sch", type=str, default='const')
# adv loss coefficient.
parser.add_argument("--adv_coef_init", type=int, default=-0) # negative
# adv loss schedule
parser.add_argument("--adv_coef_sch", type=str, default='const')
# diff loss coefficient.
parser.add_argument("--diff_coef_init", type=int, default=0) # negative
# diff loss schedule
parser.add_argument("--diff_coef_sch", type=str, default='const')

# load pretrained agent
parser.add_argument("--load", type=int, default=0)
# visualize the video
parser.add_argument("--render", type=int, default=0)
args = parser.parse_args()

# environment selection
# game env
GAME_ENV = env_list[args.env]
# random seed
GAME_SEED = args.seed
# number of game
N_GAME = args.n_games
# which victim agent to use
VIC_AGT_ID = args.vic_agt_id

# reward hyperparameters
# reward shaping parameters

# KickAndDefend
REW_SHAPE_PARAMS = {'weights': {'dense': {'reward_move': 0.5, 'reward_contact': 1, 'reward_survive': 0.5,},
                                'sparse': {'reward_remaining': 0.01}},
                   'anneal_frac': 0.01}

REW_SHAPE_PARAMS_ADV = {'weights': {'dense': {'reward_move': 0.5, 'reward_contact': 1, 'reward_survive': 0.5,},
                                'sparse': {'reward_remaining': 0.01}},
                   'anneal_frac': 0}

## sumoants
# REW_SHAPE_PARAMS = {'weights': {'dense': {'reward_move': 1}, 'sparse': {'reward_remaining': 0.01}},
#                    'anneal_frac': 0.1}

## sumohuman and you shall not pass
# REW_SHAPE_PARAMS = {'weights': {'dense': {'reward_move': 0.1}, 'sparse': {'reward_remaining': 0.01}},
#                     'anneal_frac': 0}

# reward discount factor
GAMMA = 0.99

# use victim observation
USE_VIC = False
# victim agent value network
VIC_NET = args.vic_net
# adv agent network
ADV_NET = args.adv_net

# training hyperparameters
# total training iterations.
TRAINING_ITER = 20000000
NBATCHES = 4
NEPOCHS = 4
LR = 3e-4
LR_SCHEDULE = args.lr_sch
NSTEPS = args.nsteps
CHECKPOINT_STEP = 1000000
TEST_EPISODES = 100

# loss function hyperparameters
# weight of entropy loss in the final loss
ENT_COEF = 0.00
COEF_VIC_INIT = args.vic_coef_init
COEF_VIC_SCHEDULE = args.vic_coef_sch
COEF_ADV_INIT = args.adv_coef_init
COEF_ADV_SCHEDULE = args.adv_coef_sch
COEF_DIFF_INIT = args.diff_coef_init
COEF_DIFF_SCHEDULE = args.diff_coef_sch

# callback hyperparameters
CALLBACK_KEY = 'update'
# n_env * n_steps
CALLBACK_MUL = 16384
LOG_INTERVAL = 2048

# save every 8 nupdates
CHECKPOINT_INTERVAL = 131072

PRETRAIN_TEMPLETE = "../agent-zoo/%s-pretrained-expert-1000-1000-1e-03.pkl"

# SAVE_DIR AND NAME
SAVE_DIR = '../agent-zoo/'+ GAME_ENV.split('/')[1] + '_' + str(VIC_AGT_ID)+'_' + ADV_NET + '_' + VIC_NET + '_' + \
           str(COEF_VIC_INIT) + '_' +  COEF_VIC_SCHEDULE + '_' + \
           str(COEF_ADV_INIT) + '_' +  COEF_ADV_SCHEDULE + '_' + \
           str(COEF_DIFF_INIT) + '_' + COEF_DIFF_SCHEDULE + '_' + str(USE_VIC)
EXP_NAME = str(GAME_SEED)

# choose the victim agent.
if 'You' in GAME_ENV.split('/')[1]:
    REVERSE = True
else:
    REVERSE = False


def _save(model, root_dir, save_callbacks):
    os.makedirs(root_dir, exist_ok=True)
    model_path = osp.join(root_dir, 'model.pkl')
    model.save(model_path)
    save_callbacks(root_dir)


def Adv_train(env, total_timesteps, checkpoint_interval, log_interval, callback_key, callback_mul, logger, seed, use_victim_ob):
    log_callback = lambda logger, locals, globals: env.log_callback(logger)
    # save obs-mean & variance
    save_callback = lambda root_dir: env.save_running_average(root_dir)
    last_log = 0
    last_checkpoint = 0

    def callback(locals, globals):
        nonlocal last_checkpoint, last_log
        step = locals[callback_key] * callback_mul
        if step - checkpoint_interval > last_checkpoint:
            checkpoint_dir = osp.join(out_dir, 'checkpoints', f'{step:012}')
            _save(model, checkpoint_dir, save_callback)
            last_checkpoint = step

        if step - log_interval > last_log:
            log_callback(logger, locals, globals)
            last_log = step
        return True

    model.learn(total_timesteps=total_timesteps, log_interval=1, callback=callback, seed=seed,
                use_victim_ob=use_victim_ob)


if __name__=="__main__":

        # reward_anneal decay
        scheduler = Scheduler(annealer_dict={'lr': ConstantAnnealer(LR)})
        env_name = GAME_ENV

        # multi to single, apply normalization to victim agent's observation, reward, and diff reward.
        venv = SubprocVecEnv([lambda: make_zoo_multi2single_env(env_name, VIC_AGT_ID, REW_SHAPE_PARAMS, scheduler,
                                                                reverse=REVERSE) for i in range(N_GAME)])
        # test
        if REVERSE:
            venv = Monitor(venv, 1)
        else:
            venv = Monitor(venv, 0)

        # adversarial agent reward sharping.
        rew_shape_venv = apply_reward_wrapper(single_env=venv, scheduler=scheduler,
                                              agent_idx=0, shaping_params=REW_SHAPE_PARAMS)

        # normalize adversarial agent's reward and observation.
        venv = VecNormalize(rew_shape_venv)
        # makedir output
        out_dir, logger = setup_logger(SAVE_DIR, EXP_NAME)

        if ADV_NET == 'MLP':
            adv_agent = MlpPolicy
        elif ADV_NET == 'LSTM':
            adv_agent = MlpLstmPolicy
        else:
            print('Unknow adversarial agent network type. Using mlp')
            adv_agent = MlpPolicy

        if VIC_NET == 'MLP':
            IS_MLP = True
            vic_value = MlpValue
        elif VIC_NET == 'LSTM':
            IS_MLP = False
            vic_value = MlpLstmValue
        else:
            print('Unknow victim value network type.')
            IS_MLP = True
            vic_value = MlpValue

        model = PPO_A2C(adv_agent,
                       venv,
                       lr_schedule=LR_SCHEDULE,
                       coef_opp_init=COEF_VIC_INIT,
                       coef_opp_schedule=COEF_VIC_SCHEDULE,
                       coef_adv_init=COEF_ADV_INIT,
                       coef_adv_schedule=COEF_ADV_SCHEDULE,
                       coef_abs_init=COEF_DIFF_INIT,
                       coef_abs_schedule=COEF_DIFF_SCHEDULE,
                       ent_coef=ENT_COEF,
                       nminibatches=NBATCHES, noptepochs=NEPOCHS,
                       learning_rate=LR,  verbose=1,
                       n_steps=NSTEPS, gamma=GAMMA, is_mlp=IS_MLP,
                       env_name=env_name, opp_value=vic_value, retrain_victim=False)

        Adv_train(venv, TRAINING_ITER, CHECKPOINT_INTERVAL, LOG_INTERVAL, CALLBACK_KEY, CALLBACK_MUL, logger, GAME_SEED,
                  use_victim_ob=USE_VIC)
        model.save(os.path.join(out_dir, env_name.split('/')[1]))
