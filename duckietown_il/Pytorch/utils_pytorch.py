#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 15:55:29 2020

@author: nuvilabs
"""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DuckieDataset(Dataset):

    def __init__(self, obs, actions, transform=None):
        """
        Args:
            obs: Observations array.
            actions: Array of actions.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.obs = obs
        self.actions = actions
        self.transform = transform

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):

        image = self.obs[idx]
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        

        sample = image

        if self.transform:
            sample = self.transform(sample)
        return sample, self.actions[idx]

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, image):
        
        
        image_copy = np.copy(image)

        image_copy = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #cv2.imwrite('filename.jpg', image_copy)  
        image_copy = cv2.resize(image_copy, (200,66), interpolation = cv2.INTER_AREA)
        cv2.imwrite("example_duckie.jpg",image_copy)
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0

        return image_copy

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return torch.from_numpy(image)
                