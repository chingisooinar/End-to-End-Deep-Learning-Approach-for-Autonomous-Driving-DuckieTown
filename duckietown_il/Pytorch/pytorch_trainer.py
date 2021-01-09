#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 17:16:30 2020

@author: nuvilabs
"""
import sys
sys.path.append("../")
import numpy as np
#from tqdm import tqdm



import os
import cv2
import torch



def enable_cuda(data,device):
    observations,actions = data
    return observations.float().to(device),actions.float().to(device)

class training(object):
    def __init__(self,checkpntroot,model,device,epochs,criterion,optimizer,scheduler,start_epoch,train_loader,val_loader):
        super(training,self).__init__()
        self.model=model
        self.device=device
        self.epochs=epochs
        self.check=checkpntroot
        self.criterion=criterion
        self.scheduler=scheduler
        self.optimizer=optimizer
        self.strt=start_epoch
        self.train_loader=train_loader
        self.val_loader=val_loader
    def train(self):
        min_val_loss = np.inf
        self.model.to(self.device)#gpu
        for epoch in range(self.strt,self.epochs+self.strt):
            self.scheduler.step()#decay lr if it hits milestones
            
            train_loss=0.0
            self.model.train()
            
            for i,batch in enumerate(self.train_loader):
                x, y=enable_cuda(batch,self.device)
                
                self.optimizer.zero_grad()

                
                
                prediction=self.model(x)
                #print(f'prediction : {prediction[0]} target {y[0]}')
                #print(prediction.shape,angles.shape)
                loss = (prediction - y).norm(2).mean() if self.criterion == None else self.criterion(prediction,y)#.unsqueeze(1))
                
                loss.backward()
                self.optimizer.step()
                
                train_loss+=loss.data.item()
               
                    
                if i%100==0:
                    print(f'Training loss at Epoch: {epoch} | Loss: {train_loss / (i + 1)}')
                
                #validation
                
                self.model.eval()
                
                val_loss=0.0
                
            with torch.set_grad_enabled(False):
                for i,batch in enumerate(self.val_loader):
                    x, y=enable_cuda(batch,self.device)

                    
                    prediction=self.model(x)
                #print(prediction.shape,angles.shape)
                    loss = (prediction - y).norm(2).mean() if self.criterion == None else self.criterion(prediction,y)#.unsqueeze(1))
                
                  
                
                    val_loss+=loss.data.item()
                    if i%100==0:
                        print(f'Validation Loss: {val_loss / (i + 1)}')

            

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                state = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                }

                self.save(state,"best")
            elif  epoch%5==0 or epoch==self.epochs+self.strt-1:
                min_val_loss = val_loss
                state = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                }

                self.save(state,"")
    def save(self,state,prefix):
        print("==> Save checkpoint ...")
        if not os.path.exists(self.check):
            os.makedirs(self.check)

        torch.save(state, self.check +prefix +'new-model-{}.h5'.format(state['epoch']))
                        
                        
                            
                        
                    
                    
                    
                    
                    
                    
                    
                    
                    
                
