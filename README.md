# Super Mario Bros RL
![Alt text](/asset/mario.gif)

- [x] Advantage Actor critic [[1]](#references)
- [x] Parallel Advantage Actor critic [[2]](#references)
- [x] Noisy Networks for Exploration [[3]](#references)
- [x] Proximal Policy Optimization Algorithms [[4]](#references) (WIP)
 
## 1. Setup
####  Requirements

------------

- python3.6
- [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)
- [OpenCV Python](https://pypi.python.org/pypi/opencv-python)
- [PyTorch](http://pytorch.org/)
- [tensorboardX](https://github.com/lanpa/tensorboardX)


## 2. How to Train
Modify the parameters in `mario_a2c.py` as you like.
```
python3 mario_a2c.py
```
## 3. How to Eval
Modify the `is_load_model`, `is_render` parameters in `mario_a2c.py` as you like.
```
python3 mario_a2c.py
```
## 4. Loss/Reward Graph
![image](https://user-images.githubusercontent.com/23333028/45729323-f6b9d680-bc06-11e8-9844-cc9b1433928d.png)



References
----------

[1] [Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf)    
[2] [Efficient Parallel Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1705.04862)  
[3] [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)  
[4] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)  
