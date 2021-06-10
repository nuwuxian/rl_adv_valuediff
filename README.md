# Adversarial Policy Learning in Two-player Competitive Games

This repo contains the code for the paper titiled "Adversarial Policy Learning in Two-player Competitive Games".

## Overview
We propose a novel adversarial attack against deep reinforcement learning policies in two-player competitive games. Technically, our adversarial learning method introduces a new learning algorithm that not only maximizes the expected reward of the adversarial agent but, more importantly, minimizes that of the victim.  We demonstrate that our proposed learning algorithm could effectively fail a target victim even though the corresponding game is not strictly zero-sum.

## Requirement
The requirements of each game are listed in ```requirements.txt``` under the corresponding folder. 

## Code structure and instructions 
Detailed instructions about the environment configuration and agent training can be found in ```run.md ```.

## Contact
Wenbo Guo: ```wzg13@ist.psu.edu```.

## Citation
```
W.Guo, X. Wu, S. Huang, X. Xing, "Adversarial Policy Learning in Two-player Competitive Games", In ICML 2021.
```
