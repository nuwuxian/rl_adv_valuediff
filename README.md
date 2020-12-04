# rl_reward

implementation of rl_newloss

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
  - Put ```MuJoCo/gym_compete.zip``` into ```anaconda3/envs/mujoco/lib/python3.X/site-packages/``` and run ```unzip gym_compete.zip```. You will see two folders ```gym_compete.zip``` and ```gym_compete-0.0.1.dist-info```. 
  
