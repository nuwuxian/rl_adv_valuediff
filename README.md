# Reproducing the results on the MuJoCo Game:

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

## Adv_train:
- Existing Attack: Run the ```python adv_train.py --env={env_id} --vic_coef_init=0 --adv_coef_init=-1 ```

- Our Attack: Run the  ```python adv_train.py --env={env_id} --vic_coef_init=1 --adv_coef_init=-1```

- After training is done, the trained models and tensorboard logs will be saved into the ```~/rl_newloss/MuJoCo/agent-zoo``` folder with different runs in different folders named by the starting time.

## Adv_retrain:
- Put the adversarial model used for the retraining in the ```~/rl_newloss/MuJoCo/adv_agent``` folder and name the weights of the policy network as  ```model.pkl```, the mean and variance of the observation normalization as ```obs_rms.pkl```. We have put our choice there, it can be directly used for evaluation. 

- Run the ```python victim_train.py --env={env_id} --adv_path={path-to-trained-advesaries-model} --adv_obs_normpath={path-to-trained-adversaries-obs_rms} --vic_coef_init=1 --adv_coef_init=-1 --mix_ratio=0.5``` , it will retrain the victim agent using our method.

- After training is done, the trained models and tensorboard logs will be saved into the ```~/rl_newloss/MuJoCo/victim-agent-zoo``` folder with different runs in different folders named by the starting time.

## Visualizing the winning rate of the adversarial agents or retrained victim agents:
  Run “python plot.py --log_dir XX --out_dir @@” XX refers to the path to the results, e.g., ```~/rl_newloss/MuJoCo/agent-zoo```; @@ refers to the output folder.

## Visualizing the GMM / T-SNE
  - Run ```python generate_activations.py --env={env_id} --opp_path={path-to-opponent} --vic_path={path-to-victim} --norm_path={path-to-opponent-obs_nms} --opp_type={opponent_type} --out_dir={output dir}``` to collect the victim activations when playing against different opponents. 
  
  - To generate the t-sne visualization results, run ```python plot_tsne.py --dir XX --output_dir @@```. To generate the GMM visualization results, run ```python plot_gmm.py --dir XX --output_dir @@```. XX refers to the path to the folder which save the victim activations; @@ refers to the output folder. 

# Reproducing the results on the StarCraft Game:

## Install StarCraft environment:
Refer to the ```https://github.com/Tencent/PySC2TencentExtension``` to install the StarCraft environment. Then, run the ```pip install -r requirments.txt``` to finalize the environments.

## Adv_train:
- Existing Attack: Run the ```python -m bin.advtrain_ppo --job_name learner --vic_coef_init 0 -- adv_coef_init 1 -- init_model_path '../normal-agent/checkpoint-100000' --save_dir XX &``` to start the learner. Run the ```for i in $(seq 0 20); do python -m bin.advtrain_ppo --job_name=actor --vic_coef_init 0 -- adv_coef_init 1 -- init_model_path '../normal-agent/checkpoint-100000' --save_dir XX --learner_ip local host & done;``` to start the actor. XX refers to the path to the results.

- Our Attack: Run the ```python -m bin.advtrain_ppo --job_name learner --vic_coef_init -1 -- adv_coef_init 1 -- init_model_path '../normal-agent/checkpoint-100000' --save_dir XX &``` to start the learner. Run the ```for i in $(seq 0 20); do python -m bin.advtrain_ppo --job_name=actor --vic_coef_init 0 -- adv_coef_init 1 -- init_model_path '../normal-agent/checkpoint-100000' --save_dir XX --learner_ip local host & done;``` to start the actor. XX refers to the path to the results.

## Adv_retrain:
- Modify the 51, 52, 83 line of file ```bin/adv_mixretrain_ppo``` to set the adversary agent path, norm agent path and victim path seperately.

- Run the ```python -m bin.adv_mixretrain_ppo --job_name learner &``` to start the learner. Run the 
```for i in $(seq 0 20); do python -m bin.adv_mixretrain_ppo --job_name=actor --learner_ip local host & done;```
to start the actor.
