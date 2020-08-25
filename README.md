# A3C-GS
Official Tensorflow implementation of A3C-GS from the paper "A3C-GS: Adaptive Moment Gradient Sharing With Locks for Asynchronous Actor-Critic Agents" published in IEEE Transactions on Neural Networks and Learning Systems (https://ieeexplore.ieee.org/document/9063667).

## Description
A3C-GS implements an asynchronous gradient sharing mechanism on top of parallel actor-critic algorithms to improve exploration characteristics. It has the property of automatically diversifying worker policies in the short term for exploration, thereby reducing the need for entropy loss terms. Despite policy diversification, the algorithm converges to the optimal policy in the long term. 

## Requirements
1. Python 3.7, tensorflow = 1.14, scipy, numpy, opencv-python
2. For vizdoom:
    in  ```vizdoom/__init__```, update ```_COMPILED_PYTHON_VERSION``` based on python version you are using
    
## Training
- To train A3C-GS, go inside the folder directory of desired game. For example, to train a3c-gs for doom_basic, run
```
cd doom_basic
python a3c-gs.py
```

- To train the benchmark A3C standard algorithm, go inside the folder directory of the desired game. For example, to train a3c-standard, run 
```
cd doom_basic
python a3c-standard.py
```


## Publication
 If you use this software in your research, please cite our publication:
 
```
@ARTICLE{9063667,
  author={A. B. {Labao} and M. A. M. {Martija} and P. C. {Naval}},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={A3C-GS: Adaptive Moment Gradient Sharing With Locks for Asynchronous Actor-Critic Agents}, 
  year={2020},
  volume={},
  number={},
  pages={1-15},}
  ```
