# Understanding RL within Autonomous Driving

Within the development of Autonomous driving, currently there are two major approaches, one being Reinforcement Learning (RL) and the other being Imitation Learning. This project focuses on the use of Deep Reinforcement Learning (DLR) techniques in the development of an Autonomous vehicle, in specific PPO. PPO (Proximal Policy Optimisation) is a Policy gradient method that offers an easy implementation to new environments and architectures, whilst maintaining good generalisation to unseen environments.

Due to the nature of reinforcement learning, simulators are often used to train these models, as they provide a space for the agent to experiment without real-life consequences (the safety of other actors, extra costs on repairs, etc). The use of simulators also allows to accelerate the training, as it can provide many hours of training to an agent, the hours of training would be only constrained by the computational capacity. Furthermore simulator can train agents on edge cases which helps mitigate the long tail problem, i.e. the problem of rare events that often occur in the real world.

The purpose of this project is to find a good configuration for the PPO Algorithm while evaluating its performance in the driving simulator CARLA. The main goal of the evaluation is to improve the speed of training while keeping a good generalisation performance.



## Implementation

As discuss in the previous chapters, this project will focus on the implementation of a PPO algorithm, in specific the the algorithm aims to train an agent to drive a car autonomously in a simulator environment (CARLA). Accordingly we wish to investigate different set ups to accelerate the learning of the agent as well as to keep a good performance on generalisation. In particular we will have a further look into different reward functions, the effect of visual complexity on the sensors (inputs), and the task complexity. As a result of this project we hope to demonstrate that reinforcement leaning can be used to train models in visually complex scenarios.

### Setup

CARLA is an open-source simulator for autonomous driving research, it offers an environment developed with the idea of training and validating autonomous driving systems

The recommended system requirements for the latest CARLA version are the following:

- Intel i7 gen 9th - 11th / Intel i9 gen 9th - 11th / AMD ryzen 7 / AMD ryzen 9
- +16 GB RAM memory
- NVIDIA RTX 2070 / NVIDIA RTX 2080 / NVIDIA RTX 3070, NVIDIA RTX 3080
- Ubuntu 18.04

In terms of Hardware the CARLA website recommends a modern high end PC build, with the latest components.

The system in which the model was run had a GTX 970 with 3.5GB + 0.5GB VRAM, an 8 core cpu Intel i7-4790K and 32 GB of RAM. This system worked with some limitations on expanding the model to extra sensors and features.

As part of the set up once Ubuntu 18 was installed, the installation of CARLA and Unreal Engine took about 4 hours, this was due to the size of the game engine and CARLA’s dependencies, which were about 100 GB. In addition python libraries were installed such as TensorFlow for GPU 1.13, Keras, OpenCV, Scipy, Pyglet, etc.

### Algorithm

PPO is an Actor critic algorithm, i.e PPO needs an actor <img src="https://render.githubusercontent.com/render/math?math=\pi\(a_{t}\mid%20s_{t}:\theta)#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}\pi\(a_{t}\mid%20s_{t}:\theta)#gh-dark-mode-only"> and a critic <img src="https://render.githubusercontent.com/render/math?math=V\(s_{t}:\theta_{v})#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}V\(s_{t}:\theta_{v})#gh-dark-mode-only"> network. The implementation of this architecture consists of two different multi-layer perceptrons, this is due to the input <img src="https://render.githubusercontent.com/render/math?math=s_{t}#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}s_{t}#gh-dark-mode-only"> being a vector. The actor network consists
of three layers of size 500, 300, and <img src="https://render.githubusercontent.com/render/math?math=a_{dim}#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}a_{dim}#gh-dark-mode-only">, where <img src="https://render.githubusercontent.com/render/math?math=a_{dim}#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}a_{dim}#gh-dark-mode-only"> is the size of the action space. The first two layers uses ReLU as their activation function, where as the final layer uses no activation function. This last layer represents the unscaled means of the Gaussian distributions which we sample actions from, the unscaled mean is denoted as <img src="https://render.githubusercontent.com/render/math?math=O_{i}#gh-light-mode-only"> <img src="https://render.githubusercontent.com/render/math?math=\color{White}O_{i}#gh-dark-mode-only"> and the scaled mean as <img src="https://render.githubusercontent.com/render/math?math=\mu_{i}#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}\mu_{i}#gh-dark-mode-only"> for the <img src="https://render.githubusercontent.com/render/math?math=i^{th}#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}i^{th}#gh-dark-mode-only"> action. In order to scale the output of the network to the range of each actions respective range we apply the following transformation.

<img src="https://render.githubusercontent.com/render/math?math=\mu_{i}=a_{i}^{\min%20}+\frac{\tanh%20\left(o_{i}\right)+1}{2}%20*\left(a_{i}^{\max%20}-a_{i}^{\min%20}\right)#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}\mu_{i}=a_{i}^{\min%20}+\frac{\tanh%20\left(o_{i}\right)+1}{2}%20*\left(a_{i}^{\max%20}-a_{i}^{\min%20}\right)#gh-dark-mode-only">

We pass the raw outputs of the last layer through a hyperbolic tangent function (tanh), to end up with values in the range [1, 1], and then adding 1 and diving by 2 puts our values in range of [0,1]. Subsequently we apply a linear interpolation between the <img src="https://render.githubusercontent.com/render/math?math=i^{th}#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}i^{th}#gh-dark-mode-only"> action min and max value,
which results in <img src="https://render.githubusercontent.com/render/math?math=a_{i}^{\min%20}%20\leq%20\mu_{i}%20\leq%20a_{i}^{\max%20}#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}a_{i}^{\min%20}%20\leq%20\mu_{i}%20\leq%20a_{i}^{\max%20}#gh-dark-mode-only">. This transformation is absent from the PPO paper, but it is supported by *S-RL Toolbox: Environments, Datasets and Evaluation Metrics for State Representation Learning.*

The critic network is made of three layers of sizes 500, 300, and 1, where the output is represented by <img src="https://render.githubusercontent.com/render/math?math=V\(s_{t}:\theta_{v})\approx\R\(s_{t})#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}V\(s_{t}:\theta_{v})\approx\R\(s_{t})#gh-dark-mode-only">. The first two layers uses ReLU as their activation
function, where as the final layer uses no activation function in the same way as the actor network, this allows the critic network to represent any possible value of <img src="https://render.githubusercontent.com/render/math?math=\R\(s_{t})#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}\R\(s_{t})#gh-dark-mode-only">.

When computing the clipped loss <img src="https://render.githubusercontent.com/render/math?math=L^{C%20L%20I%20P}#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}L^{C%20L%20I%20P}#gh-dark-mode-only"> of the PPO algorithm, we need to calculate the <img src="https://render.githubusercontent.com/render/math?math=r_{t}(\theta)#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}r_{t}(\theta)#gh-dark-mode-only">, with the action means and standard deviations from the network we can calculate the log probability <img src="https://render.githubusercontent.com/render/math?math=\log%20\pi_{\theta}(a%20\mid%20s)#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}\log%20\pi_{\theta}(a%20\mid%20s)#gh-dark-mode-only"> of any action a under policy given state <img src="https://render.githubusercontent.com/render/math?math=s#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}s#gh-dark-mode-only"> 

<img src="https://render.githubusercontent.com/render/math?math=\log%20\pi_{\theta}\left(a_{t}%20\mid%20s_{t}\right)-\log%20\pi_{\theta_{\text%20{old%20}}}\left(a_{t}%20\mid%20s_{t}\right)=\frac{\pi_{\theta}\left(a_{t}%20\mid%20s_{t}\right)}{\pi_{\theta_{\text%20{old%20}}}\left(a_{t}%20\mid%20s_{t}\right)}=r_{t}(\theta)#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}\log%20\pi_{\theta}\left(a_{t}%20\mid%20s_{t}\right)-\log%20\pi_{\theta_{\text%20{old%20}}}\left(a_{t}%20\mid%20s_{t}\right)=\frac{\pi_{\theta}\left(a_{t}%20\mid%20s_{t}\right)}{\pi_{\theta_{\text%20{old%20}}}\left(a_{t}%20\mid%20s_{t}\right)}=r_{t}(\theta)#gh-dark-mode-only">

Therefore once the combined loss is computed we optimise it with the Adam optimiser

