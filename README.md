# End-to-End Deep Learning Approach for Autonomous Driving: Imitation Learning

# About 🙂

In this work, I demonstrate a CNN that is indeed powerful by applying it beyond pattern recognition. Thus, it learns the entire processing pipeline required to steer a vehicle. The work is inspired by NVIDIA’s real-sized autonomous car, called DAVE-2, which drove on public roads autonomously while only relying on the CNN. Therefore, the identical architecture is implemented and tested in various environments :)

# Paper

End to End Learning for Self-Driving Cars

- [https://arxiv.org/pdf/1604.07316v1.pdf](https://arxiv.org/pdf/1604.07316v1.pdf)
# Read More
End-to-End Deep Learning Approach for Autonomous Driving: Imitation Learning
https://chingisoinar.medium.com/end-to-end-deep-learning-approach-for-autonomous-driving-imitation-learning-f215d534715c

# **Training**

![End-to-End%20Deep%20Learning%20Approach%20for%20Autonomous%20D%203b36f497db714575b21d3fd89fd7e53b/Untitled.png](End-to-End%20Deep%20Learning%20Approach%20for%20Autonomous%20D%203b36f497db714575b21d3fd89fd7e53b/Untitled.png)

The figure provides a high-level overview of how NVIDIA collected data in order to train the proposed model. The DAVE-2 was trained on hours of human driving in similar, but not identical, environments. The training data included video from three cameras and the steering commands sent by a human operator. The training data is augmented with additional images that display the car in various positions from the middle of the lane and rotations from the direction of the road. Therefore, the shifted images are obtained from two additional cameras placed on both sides of the central camera. So, as the image passed into the CNN-based model, a network computed steering command is generated which is compared to a desired steering command obtained from a driver.

# Architecture

![End-to-End%20Deep%20Learning%20Approach%20for%20Autonomous%20D%203b36f497db714575b21d3fd89fd7e53b/Untitled%201.png](End-to-End%20Deep%20Learning%20Approach%20for%20Autonomous%20D%203b36f497db714575b21d3fd89fd7e53b/Untitled%201.png)

The figure above demonstrates the overall architecture used as a model. The weights of our network are trained to minimize the mean squared error between the steering command output by the network and the actual command provided. The network contains 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers. The initial layer of the network performs image normalization. The normalizer is manually coded and cannot be adjusted while the learning process. Performing normalization in the network helps the normalization scheme to be changed with the network architecture and to be boosted using GPU processing. The convolutional layers were designed to perform feature extraction. The strided convolutions are used in the first three convolutional layers with a 2×2 stride and a 5×5 kernel and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers.The five convolutional layers are followed by three fully connected layers leading to an output control value which is the inverse turning radius.

# Simalutor

![End-to-End%20Deep%20Learning%20Approach%20for%20Autonomous%20D%203b36f497db714575b21d3fd89fd7e53b/Untitled%202.png](End-to-End%20Deep%20Learning%20Approach%20for%20Autonomous%20D%203b36f497db714575b21d3fd89fd7e53b/Untitled%202.png)

Duckietown is an open, inexpensive and flexible platform used for autonomy education and research. Originally, the simulator was focused on robotics; however, it is now widely used for autonomous driving simulation as well. The platform comprises small autonomous robots, which are called as *Duckiebots*, built from off-the-shelf components, and cities that contain a large variety of objects, including obstacles, traffic lights, and citizens. Therefore, the simulator is used not only for lane navigation tasks but equally for other problems such as object detection and recognition. The platform offers a wide range of functionalities at a low cost.

Duckietown simulator is used for training Reinforcement Learning agents and performing various tasks. The observations are single camera images of the size 120 by 160.

> Get Started: [DuckieTown](https://www.duckietown.org/)

# Get Started

- duckietown_il/ - contains scripts for training implemented in both Pytorch and Keras, located in Pytorch/ and Keras/ respecitvely.
- In  duckietown_il/Pytorch/pytorch_model.py there are two models; however, CNNcar is the proposed DAVE-2 architecture.
- logger/log.py is a script for getting observation-action pair of data while testing a RL agent.
