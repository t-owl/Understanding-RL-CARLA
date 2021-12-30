# Understanding RL within Autonomous Driving

Within the development of Autonomous driving, currently there are two major approaches, one being Reinforcement Learning (RL) and the other being Imitation Learning. This project focuses on the use of Deep Reinforcement Learning (DLR) techniques in the development of an Autonomous vehicle, in specific PPO. PPO (Proximal Policy Optimisation) is a Policy gradient method that offers an easy implementation to new environments and architectures, whilst maintaining good generalisation to unseen environments.

Due to the nature of reinforcement learning, simulators are often used to train these models, as they provide a space for the agent to experiment without real-life consequences (the safety of other actors, extra costs on repairs, etc). The use of simulators also allows to accelerate the training, as it can provide many hours of training to an agent, the hours of training would be only constrained by the computational capacity. Furthermore simulator can train agents on edge cases which helps mitigate the long tail problem, i.e. the problem of rare events that often occur in the real world.

The purpose of this project is to find a good configuration for the PPO Algorithm while evaluating its performance in the driving simulator CARLA. The main goal of the evaluation is to improve the speed of training while keeping a good generalisation performance.

## Requirements

System requirements:

- Ubuntu 18.04
- 100 GB disk space
- An adequate GPU (with at least 4 GB VRAM)
- For further instructions on the prerequisites follow the guide described in the CARLA documentation. https://carla.readthedocs.io/en/0.9.11/build_linux/

Software requirements:

- Python 3.6
- CARLA 0.95 or above
- TensorFlow for GPU
- OpenAI gym
- OpenCV for Python

## Running

Within the code folder there are already 3 pretrained PPO agents: `PPO_Junior` , ` PPO_Junior-sun`, `PPO_Junior_semantic`, they are located in the `models` folder. The best performing model is the ` PPO_Junior_semantic` model.

There are two modes of operation training and running.

**Training:**

To train a new model, from the root directory you need to run the following command

```shell
python3 train.py --model_name name_of_your_model -start_carla
```

this will create a new folder inside the `models` folder, which will store the model data

**Running:**

Once everything is installed you can run the models described before by running the following command.

```shell
python3 run_eval.py --model_name PPO_Junior_semantic -start_carla
```

This will run an evaluation of the model and you will be able to test the running model.

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

### Reward Functions

In order to try different reward functions, there is a need for metrics to measure performance in between reward functions, the metrics recorded were the total distance travelled in meters, the number of laps completed, total and average deviance from the centre of the lane, and the average speed.

**Target speed**. The first reward function, acts on a target speed, which aims to train the model close to the target speed, by having a <img src="https://render.githubusercontent.com/render/math?math=v_{\text%20{norm%20}}=\frac{v}{v_{\text%20{target%20}}}#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}v_{\text%20{norm%20}}=\frac{v}{v_{\text%20{target%20}}}#gh-dark-mode-only"> we make sure to obtain <img src="https://render.githubusercontent.com/render/math?math=v_{\text%20{norm%20}}=1#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}v_{\text%20{norm%20}}=1#gh-dark-mode-only"> when <img src="https://render.githubusercontent.com/render/math?math=v=v_{\text%20{target%20}}#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}v=v_{\text%20{target%20}}#gh-dark-mode-only">, also <img src="https://render.githubusercontent.com/render/math?math=v_{\text%20{norm%20}}#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}v_{\text%20{norm%20}}#gh-dark-mode-only"> will shrink linearly from 1 to 0 when the car gets away from the target speed. this reward function will encourage the car to stay close to the target speed. Additionally the car will stop on any other infraction (going off the road, having a collision, etc) and will give a reward of -10 to deter the agent going into these state.

**Keep Centred**. By taking advantage of the simulator utilities, we can then use the measurements of distance between the centre of the car and the centre of the lane. This reward function builds upon the previous one, and adds a function of the distance to the centre of the lane which is described by <img src="https://render.githubusercontent.com/render/math?math=1-d_{\text%20{norm%20}},%20d_{\text%20{norm%20}}=\frac{d}{d_{\max%20}}#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}1-d_{\text%20{norm%20}},%20d_{\text%20{norm%20}}=\frac{d}{d_{\max%20}}#gh-dark-mode-only"> this function is inversely proportional to the distance <img src="https://render.githubusercontent.com/render/math?math=d#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}d#gh-dark-mode-only">. The idea behind this reward function is that we want our agent to minimise the
distance between the centre of the lane and our car while maintaining a speed close to the target speed.

**Aligned with the road**. While working with the other reward functions it was clear that the cart struggled to be aligned with the road and had erratic steering behaviour, by adding this to the reward function we hope to discourage this behaviour.

In this reward function we introduced the angel term <img src="https://render.githubusercontent.com/render/math?math=\alpha_{\text%20{rew%20}}#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}\alpha_{\text%20{rew%20}}#gh-dark-mode-only">, this represents the angle difference between the vehicle’s forward vector, and the current way point’s forward vector.

### Evaluation

The way the model was evaluated was by creating certain conditions in order to have good understanding of the performance of the model. The conditions consisted of the completion of 3 laps to a given circuit, this would prove that the agent had understood and generalised the knowledge gather through training. As part of the evaluation algorithm checkpoints were added, in other words every time the car had an infraction, the car would start from a checkpoint set by the algorithm instead of starting from the start. This meant more time would be spend trying to overcome any problems found through the circuit.

### RGB To Semantic Segmentation Sensors

During the design stage of the project different sensors were consider as apart of the implementation, e.g. RGB cameras, Semantic Segmentation Sensors, Radars, Lidars, etc. To start the implementation with a regular RGB camera was attached to the front of the car. On this first run the reward function used was the combination of all reward functions describe previously, the training lasted for 5 hours approximately completing a total of 850 episodes. In the first 40 episodes, 15 min the algorithm managed to get to the first 250 meters, which happen to be the location of the first turn of the circuit. From episode 40 to 180 the model was trying to breakthrough the first turn by using the checkpoint next to the turn thus mainly focusing on this turn during this episodes the car managed to reach the first turn (250 meters) during the recording of each evaluation (every 10 episodes). After 180 episodes the model had a clear degeneration, where the car had trouble reaching the first turn on the evaluation runs, going as far as only reaching 26 meters on average. After five hours of training the car farthest distance was 250 meters. As a consequence of these results we came to the conclusion as a hypothesis that this behaviour was due to the many attempts at turning while starting from the checkpoint, this meant that a generalised knowledge was built at these attempts, and when starting from the beginning of the track the car used this generalised knowledge and failed to go further.

During the training we observed that in the very first turn the car seemed to avoid a shadow and that would have been the reason the car could not pass the first turn. This obstacle is shown in the following picture.

![Pasted image 20210805161943](https://user-images.githubusercontent.com/96732103/147761046-f1ab5d91-3645-4f11-ba3b-4ed0d9f1dabb.png)


Once we came to the previous conclusion a new model was created to test the hypothesis of the shadows affecting the training. This model was trained with the same reward function the only difference was the modification of the sun angle, this was done with the intention of avoiding the shadow projected on the floor of the first turn. As a result of this change the car managed to breakthrough the first turn within the first 200 episodes, clearly indicating the shadows had an effect on the training.

Finally, after the previous realisation a model based on a Semantic Segmentation Sensors was created. The main difference was the sensor in the front of the vehicle (RGB To Semantic Segmentation Sensors). Due to the nature of Semantic Segmentation, shadows are to be ignored, therefore the initial problem we had on the first model would be technically not affect this model.

The image segmentation model outperform with a great difference the other two models. Within the first 20 minutes (40 episodes) the car learned how to turn the first corner, in contrast the first model never learned how to turn after 5 hours of training

The total training time of this model was 21 hours. In the the first 11 hours (590 episodes) the car learned the different type of obstacles and turns in the circuit and was able to complete over two laps, but as mentioned before the full evaluation consisted of 3 entire laps, therefore within the next 10 hours after the two laps the car was able to learn the circuit and also to complete the full 3 laps.

### Conclusion

Subsequently, after working in the implementation there was one main observations of the experiment that summarises the performance and evaluation of the different models. This was a huge performance boost that was obtained with the semantic segmentation sensor compared to the RGB one. This performance boost was consistent for both tasks of the experiment, training and evaluation. It is clear that the models with the semantic segmentation sensor were able to identify very well all the main patterns of the images collected. On the contrary, the RGB sensors was very unreliable. In most cases the model was not able to understand shadows from the live images, and therefore it was impossible to generalise knowledge to unseen data as there was not enough training data. By directly comparing the data extracted on the evaluation of each model, we can see that the average distance for the RGB model was 44.35 meters compared to the semantic model which was 357.20 meters. we can see the overall performance on both models in the following graph.

![image](https://user-images.githubusercontent.com/96732103/147761107-086b70d1-74a4-42be-93fd-167e5cb0d2f7.png)


The difference in performance helped even further the semantic segmentation model as it helped find generalisation to different scenarios where as on the contrary the RGB model was hindered by generalising bad decisions on the same repetitive task trying to avoid the shadow present in the first turn of the experiment.
