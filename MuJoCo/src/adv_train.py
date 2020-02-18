import os
import argparse
import gym
from common import env_list
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from scheduling import ConstantAnnealer, Scheduler
from shaping_wrappers import apply_reward_wrapper
from environment import make_zoo_multi2single_env, Monitor
from logger import setup_logger
from ppo2_wrap import MyPPO2
from value import MlpValue, MlpLstmValue
# from common import get_zoo_path


model_dir = '../agent-zoo'
adv_rew_shape_params = {'weights': {'dense': {'reward_move': 0.1}, 'sparse': {'reward_remaining': 0.01}},
                        'anneal_frac': 0}
vic_rew_shape_params = {'weights': {'dense': {'reward_move': 0.1}, 'sparse': {'reward_remaining': 0.01}},
                        'anneal_frac': 0}

gamma = 0.99
training_iter = 20000000
ent_coef = 0.00
nminibatches = 4
noptepochs = 4
learning_rate = 3e-4
# lstm length should be small
n_steps = 2048
checkpoint_step = 1000000
test_episodes = 100
lr_schedule = 'const'

coef_opp_init = -1
coef_opp_schedule = 'const'
coef_adv_init = 1
coef_adv_schedule = 'const'
coef_abs_init = -1
coef_abs_schedule = 'const'


callback_key = 'update'
callback_mul = 16384
log_interval = 2048

# number of copied game environment.
n_cpu = 8
pretrain_template = "../agent-zoo/%s-pretrained-expert-1000-1000-1e-03.pkl"


def Adv_train(env, total_timesteps, callback_key, callback_mul, logger):
    # log_callback
    log_callback = lambda logger, locals, globals: env.log_callback(logger)

    last_log = 0

    def callback(locals, globals):
        nonlocal last_log
        step = locals[callback_key] * callback_mul
        if step - log_interval > last_log:
            log_callback(logger, locals, globals)
            last_log = step

        return True

    model.learn(total_timesteps=total_timesteps, log_interval=1, callback=callback)

if __name__=="__main__":

        parser = argparse.ArgumentParser()
        # game env
        parser.add_argument("--env", type=int, default=5)
        parser.add_argument("--seed", type=int, default=0)
        # load pretrained agent
        parser.add_argument("--load", type=int, default=0)
        # visualize the video
        parser.add_argument("--render", type=int, default=0)
        # which agent to attack
        parser.add_argument("--reverse", type=int, default=0)
        # output dir
        parser.add_argument('--root_dir', type=str, default="../agent-zoo")
        parser.add_argument('--exp_name', type=str, default="ppo2")
        args = parser.parse_args()

        # anneal decay
        scheduler = Scheduler(annealer_dict={'lr': ConstantAnnealer(learning_rate)})
        env_name = env_list[args.env]

        # create a env.
        env = gym.make(env_name)

        # multi to single
        venv = SubprocVecEnv([lambda: make_zoo_multi2single_env(env_name, vic_rew_shape_params, scheduler, reverse=False)
                              for i in range(n_cpu)])
        # test
        venv = Monitor(venv, 0)

        # reward sharping.
        rew_shape_venv = apply_reward_wrapper(single_env=venv, scheduler=scheduler,
                                              agent_idx=0, shaping_params=adv_rew_shape_params)
        # normalize reward
        venv = VecNormalize(rew_shape_venv)
        # makedir output
        out_dir, logger = setup_logger(args.root_dir, args.exp_name)

        model = MyPPO2(MlpPolicy,
                       venv,
                       lr_schedule=lr_schedule,
                       coef_opp_init=coef_opp_init,
                       coef_opp_schedule=coef_opp_schedule,
                       coef_adv_init=coef_adv_init,
                       coef_adv_schedule=coef_adv_schedule,
                       coef_abs_init=coef_abs_init,
                       coef_abs_schedule=coef_abs_schedule,
                       ent_coef=ent_coef,
                       nminibatches=nminibatches, noptepochs=noptepochs,
                       learning_rate=learning_rate,  verbose=1,
                       n_steps=n_steps, gamma=gamma, is_mlp=False,
                       env_name=env_name, opp_value=MlpLstmValue) # , rl_path=rl_path, var_path=var_path)
        '''
        if args.load == 0:
            model = PPO2(MlpPolicy, 
                         venv, 
                         ent_coef=ent_coef,
                         nminibatches=nminibatches,
                         noptepochs=noptepochs,
                         learning_rate=learning_rate,
                         verbose=1,
                         n_steps=n_steps,
                         gamma=gamma)
                         # seed=args.seed,
                         # n_cpu_tf_sess=1)
        else:
            model = PPO2.load(pretrain_template%(env_name.split("/")[1]), 
                              venv, 
                              gamma=gamma,
                              ent_coef=ent_coef, 
                              nminibatches=nminibatches, 
                              learning_rate=learning_rate,
                              n_steps=n_steps)
        '''
        Adv_train(venv, training_iter, callback_key, callback_mul, logger)
        model.save(os.path.join(out_dir, env_name.split('/')[1]))
