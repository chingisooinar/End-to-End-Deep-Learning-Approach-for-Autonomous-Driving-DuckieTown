#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:34:20 2020

@author: nuvilabs
"""

import sys
sys.path.append("../")
import numpy as np
#from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
import argparse
from _loggers import Reader
from sklearn.model_selection import train_test_split
from pytorch_trainer import training,enable_cuda
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from pytorch_model import CNNcar,Model
from torch.utils import data
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from utils_pytorch import Normalize,ToTensor, DuckieDataset

def numpy_ewma_vectorized_v2(data, window):

    alpha = 2 /(window + 1.0)

    alpha_rev = 1-alpha

    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]

    offset = data[0]*pows[1:]

    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr

    cumsums = mult.cumsum()

    out = offset + cumsums*scale_arr[::-1]

    return out

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data",default='train-102k', type=str, help="name of the data to learn from (without .log)")
ap.add_argument("-e", "--epoch",default=80, type=int, help="number of epochs")
ap.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
args = vars(ap.parse_args())
DATA = args["data"]
ckptroot='./trained_models/'
# configuration zone
BATCH_SIZE = args["batch_size"]        # define the batch size
EPOCHS     = args["epoch"]             # how many times we iterate through our data
STORAGE_LOCATION = "trained_models/"   # where we store our trained models
reader = Reader(f'./train-102k.log')      # where our data lies
MODEL_NAME = "01_NVIDIA"

observations, actions = reader.read()  # read the observations & actions from data
actions = np.array(actions)            # convert actions to a np array
observations = np.array(observations)  # convert observations to a np array
x_train, x_val, y_train, y_val = train_test_split(observations, actions, test_size=0.05, random_state=2)

    
transformations = transforms.Compose([Normalize(), ToTensor()])

cv2.imwrite("example_duckie.jpg",x_train[15])

train_set =  DuckieDataset(x_train,y_train,transformations)#dataset_obj(x_train,y_train,transformations) 
val_set = DuckieDataset(x_val,y_val,transformations) 
train_loader = DataLoader(train_set, batch_size = 32, shuffle = True)
val_loader = DataLoader(val_set, batch_size = 32, shuffle = False)



cnn_model =Model(action_dim=2, max_action=1.0)
#cnn_model = CNNcar()
#cnn_model.load_state_dict(torch.load('./trained_models/both-nvidia-model-80.h5')['state_dict'])


optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, weight_decay=1e-4, momentum=0.9)

criterion = nn.MSELoss()
scheduler=MultiStepLR(optimizer,milestones=[30,60],gamma=0.1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_model.cuda()
trainer=training(ckptroot,cnn_model,device,EPOCHS,criterion,optimizer,scheduler,0,train_loader,val_loader)
trainer.train()
 
