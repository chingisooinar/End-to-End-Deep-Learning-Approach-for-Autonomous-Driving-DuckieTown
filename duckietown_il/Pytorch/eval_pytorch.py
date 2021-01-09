#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 19:17:46 2020

@author: nuvilabs
"""
import imageio
import sys
sys.path.append("../../")
from duckietown_rl.gym_duckietown.simulator import Simulator
import cv2
import torch
import numpy as np
from pytorch_model import CNNcar,Model
from utils_pytorch import Normalize,ToTensor, DuckieDataset
env = Simulator(seed=123, map_name="zigzag_dists", max_steps=5000001, domain_rand=True, camera_width=640,
                camera_height=480, accept_start_angle_deg=4, full_transparency=True, distortion=True,
                randomize_maps_on_reset=False, draw_curve=False, draw_bbox=False, frame_skip=1, draw_DDPG_features=False)

# TODO: Put your model name here!
model = Model(action_dim=2, max_action=1.0)#CNNcar()
model.load_state_dict(torch.load("path-here")['state_dict'])
model = model.double()
observation = env.reset()
env.render()
cumulative_reward = 0.0
EPISODES = 20
STEPS = 1000
model.eval()
images=[]
all_rewards = []
for episode in range(0, EPISODES):
    rewards = []
    for steps in range(0, STEPS):
        # Cut the horizon: obs.shape = (480,640,3) --> (300,640,3)
        observation = np.array(observation[150:450, :])
        # we can resize the image here
        #observation = cv2.resize(observation, (120, 60))
        # NOTICE: OpenCV changes the order of the channels !!!
        images.append(cv2.cvtColor(observation, cv2.COLOR_BGR2RGB))
        if len(images)==500:
            print('saving')
            imageio.mimsave('./movie1.gif', images) #save gif
            images=[]
            

        observation = Normalize()(observation)
        observation = ToTensor()(observation)
        observation=torch.unsqueeze(observation,1)
        with torch.set_grad_enabled(False):
            action = model(observation)

        observation, reward, done, info = env.step(action[0])
        cumulative_reward += reward

        rewards.append(reward)
        #print(reward)
        if done:
            env.reset()
            #print(f"DONE! after {steps}/{STEPS} steps!")
            break



        env.render()
    all_rewards += rewards
    env.reset()

print('total reward: {}, mean reward: {}'.format(cumulative_reward, np.mean(rewards)))

env.close()
