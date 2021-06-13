# MuJoCo Games:

## Install Mujoco environment: 
  - Install conda3 on your machine (https://www.anaconda.com/products/individual);  
  - Run ```conda create -n mujoco python==3.6``` to create a virtual environment (python 3.7 also works if python 3.6 is not available);  
  - Run ```conda activate mujoco``` to activate this environment;  
  - Run ```pip install -U scikit-learn``` to install scikit learn;  
  - Run ```pip install tensorflow==1.14``` to install the tensorflow;  
  - Run ```sudo apt-get update && sudo apt-get install cmake libopenmpi-dev zlib1g-dev``` to install the openmpi;  
  - Run ```pip install git+git://github.com/HumanCompatibleAI/baselines.git@f70377#egg=baselines```
  - Run ```pip install git+git://github.com/HumanCompatibleAI/baselines.git@906d83#egg=stable-baselines```
  - Run ```pip install mujoco-py==0.5.7```
  - Run ```pip install git+git://github.com/HumanCompatibleAI/gym.git@1918002#wheel=gym``` (Note that you will encounter an error about a conflict of the required version of the gym. Please just ignore this error. It wouldn't influence the running. )
  - Put ```MuJoCo/gym_compete.zip``` into ```anaconda3/envs/mujoco/lib/python3.X/site-packages/``` and run ```unzip gym_compete.zip```. You will see two folders ```gym_compete``` and ```gym_compete-0.0.1.dist-info```.

## Training adversarial policies:
- Existing Attack: Run the ```python adv_train.py --env <env_id> --vic_agt_id <vic_agt_id> --vic_coef_init 0 --adv_coef_init -1 ```. 

- Our Attack: Run the  ```python adv_train.py --env <env_id> --vic_agt_id <vic_agt_id>  --vic_coef_init 1 --adv_coef_init -1```.

- Attack that does not have monotonic property: Run the  ```python adv_a2ctrain.py --env <env_id> --vic_agt_id <vic_agt_id>  --vic_coef_init 1 --adv_coef_init -1```

  'env' specifies the game environment, 'vic_agt_id' specifies the victim policy under attacking (The exact choices for each game are shown in ```adv_train.py'''). ```adv_train.py''' also gives the descriptions and default values for other hyper-parameters. 

- After training is done, the trained models and tensorboard logs will be saved into the ```~/rl_newloss/MuJoCo/agent-zoo/XXXX```, where 'XXXX' is the folder name. This name is the starting time of running the current attack.

## Robustifying victim agents:
- Put the adversarial model used for the retraining in the ```~/rl_newloss/MuJoCo/adv_agent``` folder and name the weights of the policy network as  ```model.pkl```, the mean and variance of the observation normalization as ```obs_rms.pkl```. This folder currently contains the adversarial agents used for the retraining experiments in our evaluation.

- Run the ```python victim_train.py --env <env_id> --vic_agt_id <vic_agt_id> --adv_path <path-to-trained-advesaries-model> --adv_obs_normpath <path-to-trained-adversaries-observation-normalization> --vic_coef_init 1 --adv_coef_init -1 --mix_ratio <ratio of adversarial trajectories>``` , it retrains the victim agent using our method. Note that, the choice of 'vic_agt_id' should be kept the same with that in 'adv_train.py' 

- After training is done, the trained models and tensorboard logs will be saved into the ```~/rl_newloss/MuJoCo/victim-agent-zoo/XXXX```, where 'XXXX' is the folder name. This name is the starting time of running the current attack.

## Evaluation - playing two agents and recording the game results:
- Run ```python eval.py --env <env_id> --pi0_type <use-a-policy-trained-with-our-code-or-the-zoo-agents-given-in-multiagent-competition-for-party-0> --pi0_nn_type <the-policy-network-type-for-party-0-mlp-or-lstm> --pi0_path <the-path-to-the-policy-network-of-party-0> --pi0_norm_path <the-path-to-the-observation-normalization-of-party-0> --pi1_type <use-a-policy-trained-with-our-code-or-the-zoo-agents-given-in-multiagent-competition-for-party-1> --pi1_nn_type <the-policy-network-type-for-party-1-mlp-or-lstm> --pi1_path <the-path-to-the-policy-network-of-party-1> --pi1_norm_path <the-path-to-the-observation-normalization-of-party-1>```.

Playing an opponent agent with a masked victim: 
- Run ```python test_masked_victim.py --env <env_id> --opp_path <path-to-the-opponent-model> --norm_path <path-to-the-opponent-observation-normalization> --vic_path <path-to-the-victim-model>  --vic_mask <True>``` to let the opponents play with the masked victims.


## Visualizing the winning rate of the adversarial agents / retrained victim agents:
  - Run ```python plot.py --log_dir <path to the adversary attack results> --out_dir <output folder>```.
  
  - Run ```python retrain_plot.py --log_dir <path to the robustifying results> --out_dir <output folder>``` XX refers to the path to the adversary retraining results, e.g., ```~/rl_newloss/MuJoCo/victim-agent-zoo```; @@ refers to the output folder.
  

## Visualizing the GMM average likelihood / t-SNE:
  - Run ```python generate_activations.py --env <env_id> --opp_path <path-to-the-opponent-model> --vic_path <path-to-the-victim-model> --norm_path <path-to-the-opponent-observation-normalizations> --opp_type <type-of-the-opponent> --out_dir <output folder>``` to collect the victim activations when playing against different opponents. 
  
  - To generate the t-SNE visualization results, run ```python plot_tsne.py --dir <path-to-victim-activations> --output_dir <output-folder>```. To generate the GMM visualization results, run ```python plot_gmm.py --dir <path-to-victim-activations> --output_dir <output-folder>```.


## Comparison with another recently published attack on MuJoCo games:
We also compare our proposed attack with another recently published attack and show the results in the following doc: (https://docs.google.com/document/d/1YHfKyw2uCYdvVjEDmgSUW0xY4wt9rO2eJt_AwcknBkk/edit?usp=sharing).

# StarCraft II Game:

## Install StarCraft environment:
First, installing the StarCraft environment by pulling and installing the repo from: https://github.com/Tencent/PySC2TencentExtension. Then, run the ```pip install -r requirments.txt``` to install the required packages.

## Training adversarial agents:
Existing Attack: Run the ```python -m bin.advtrain_ppo --job_name learner --vic_coef_init 0 -- adv_coef_init 1 -- init_model_path '../normal-agent/checkpoint-100000' --save_dir <path-to-the-folder-that-saves-the-results> &``` to start the learner. To start the actor, write the following insturctions into a '.sh' file and run it. Note that '20' refers to the number of actors.
```
for i in $(seq 0 20); 
    do python -m bin.advtrain_ppo --job_name=actor --vic_coef_init 0 -- adv_coef_init 1 \
       -- init_model_path '../normal-agent/checkpoint-100000' --save_dir XX --learner_ip localhost & 
done;
``` 

Attack that does not have monotonic property: Run the ```python -m bin.advtrain_a2c --job_name learner --vic_coef_init -1 -- adv_coef_init 1 -- init_model_path '../normal-agent/checkpoint-100000' --save_dir <path-to-the-folder-that-saves-the-results> &``` to start the learner. To start the actor, write the following insturctions into a '.sh' file and run it. Note that '20' refers to the number of actors.

``` 
for i in $(seq 0 20); 
    do python -m bin.advtrain_a2c --job_name=actor --vic_coef_init -1 -- adv_coef_init 1 \
       -- init_model_path '../normal-agent/checkpoint-100000' --save_dir XX --learner_ip localhost & 
done;
``` 

Our Attack: Run the ```python -m bin.advtrain_ppo --job_name learner --vic_coef_init -1 -- adv_coef_init 1 -- init_model_path '../normal-agent/checkpoint-100000' --save_dir <path-to-the-folder-that-saves-the-results> &``` to start the learner. To start the actor, write the following insturctions into a '.sh' file and run
it. Note that '20' refers to the number of actors.

``` 
for i in $(seq 0 20); 
    do python -m bin.advtrain_ppo --job_name=actor --vic_coef_init -1 -- adv_coef_init 1 \
       -- init_model_path '../normal-agent/checkpoint-100000' --save_dir XX --learner_ip localhost & 
done;
``` 

## Robustifying victim agents:
Modify the 51, 52, 83 lines of file ```bin/adv_mixretrain_ppo.py``` to set the adversarial agent path, norm agent path, and victim path separately.

Run the ```python -m bin.adv_mixretrain_ppo --job_name learner --save_dir XX &``` to start the learner. To start the actor, write the following insturctions into a '.sh' file and run it. 

```  
for i in $(seq 0 20); 
    do python -m bin.adv_mixretrain_ppo --job_name=actor --save_dir XX --learner_ip localhost & 
done;
```

## Evaluation:
Run ```python -m bin.evaluate_vs_rl.py --model_path=<path-of-the-opponent-model> --victim_path=<path-of-the-victim-model> --mask_victim=False``` to play against an opponent with a victim.

Playing an opponent agent with a masked victim:  
Run ```python -m bin.evaluate_vs_rl.py --model_path=<path-of-the-opponent-model> --victim_path=<path-of-the-victim-model> --mask_victim=True``` to play against an opponent with a masked victim.

## Visualizing the winning rate of the adversarial agents or retrained victim agents:
- Run ```python plot.py --log_dir <path to the adversary attack results> --out_dir <output folder>```.
- Run ```python plot_victim.py --log_dir <path to the robustifying results> --out_dir <output folder>```.

## Visualizing the GMM average likelihood / t-SNE:
  - Run ``` python -m bin.generate_activations.py --model_path=<path-of-the-opponent-model> --model_type=<type-of-the-opponent> --victim_path=<path-of-the-victim-model> --out_path=<output folder>``` to collect the victim activations when playing against different opponents.
  
  - To generate the t-SNE visualization results, run ```python plot_tsne.py --dir <path-to-victim-activations> --output_dir <output-folder>```. To generate the GMM visualization results, run ```python plot_gmm.py --dir <path-to-victim-activations> --output_dir <output-folder>```.
