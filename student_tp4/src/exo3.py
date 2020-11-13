import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from torch.utils.data import Dataset, DataLoader
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

#  TODO:  Question 3 : Prédiction de séries temporelles


BATCH_SIZE = 100
HIDDEN_SIZE = 10
OUTPUT_SIZE = 1
    
path = "C:/Users/wuwen/Desktop/EtudeSup/Git/AMAL/student_tp4/data/tempAMAL_train.csv"
path_test = "C:/Users/wuwen/Desktop/EtudeSup/Git/AMAL/student_tp4/data/tempAMAL_test.csv"

X_train, y_train = getData(path, task="forecasting_")
X_test, y_test = getData(path_test, task="forecasting_")

writer = SummaryWriter()
h0 = torch.randn((BATCH_SIZE, HIDDEN_SIZE), dtype=torch.double)

print("Shape 1 : ", X_train.shape, y_train.shape)


data_train = DataLoader(MonDataset(X_train, y_train), shuffle=True, batch_size=BATCH_SIZE)
data_test = DataLoader(MonDataset(X_test, y_test), shuffle=True, batch_size=BATCH_SIZE)

model = RNN(BATCH_SIZE, 1, HIDDEN_SIZE, OUTPUT_SIZE, task="forecasting")
model = model.double()
#model = model.to(device)

criterion = torch.nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr = 0.01)

loss_train = []
loss_test= []
acc_train = []
acc_test = []

for x,y in data_train:
    print("sss : ", x.shape, y.shape)
    break

for i in tqdm(range(50)):
    loss_val2 = 0
    loss_val = 0
    acc_val = 0
    acc_val2 = 0
    c = 0
    c2 = 0

    for x,_ in data_train:
        model.train()
        with torch.autograd.set_detect_anomaly(False):
            optim.zero_grad()
            c += 1
            x = x.double()
            #print("x : ", x.shape)
            h, y, all_h_dec = model(x.permute(1,0,2), h0)
            
            #print("22 m ", all_h_dec.shape, x.permute(1,0,2).shape)
       
            loss = criterion(all_h_dec, x.permute(1,0,2))
            loss.backward(retain_graph=True)
            loss_val += loss.item()
            
            
            optim.step()
            
        loss_train.append(loss_val / c)
        #loss_test.append(loss_val2 / c2)
        writer.add_scalars('Temp/RNN/MSE/Loss', {'train' : loss_val / c}, i)          


"""
for city in range(len(X_train)):

    data_train = DataLoader(MonDataset(X_train[city], y_train[city]), shuffle=True, batch_size=BATCH_SIZE)
    data_test = DataLoader(MonDataset(X_test[city], y_test[city]), shuffle=True, batch_size=BATCH_SIZE)
    
    model = RNN(BATCH_SIZE, 1, HIDDEN_SIZE, OUTPUT_SIZE, task="forecasting")
    model = model.double()
    #model = model.to(device)
    
    criterion = torch.nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr = 0.1)
    
    loss_train = []
    loss_test= []
    acc_train = []
    acc_test = []
    
    for i in tqdm(range(100)):
        loss_val2 = 0
        loss_val = 0
        acc_val = 0
        acc_val2 = 0
        c = 0
        c2 = 0
    
        for x,_ in data_train:
            model.train()
            with torch.autograd.set_detect_anomaly(False):
                optim.zero_grad()
                c += 1
                x = x.double()
                h, y, all_h_dec = model(x.permute(1,0,2), h0)
                
                #print("22 m ", all_h_dec.shape, x.permute(1,0,2).shape)
           
                loss = criterion(all_h_dec, x.permute(1,0,2))
                loss.backward(retain_graph=True)
                loss_val += loss.item()
                
                
                optim.step()
          
        
        for x, _ in data_test:
            with torch.no_grad():
                model.eval()
                c2 += 1
                pred, dec = model(x.permute(1,0,2), h0)
                
                loss = criterion(dec, x.permute(1,0,2))
                loss_val2 += loss.item()

         
        #print("Val : ", loss_val2 / c2)
        loss_train.append(loss_val / c)
        #loss_test.append(loss_val2 / c2)
        writer.add_scalars('Temp/RNN/MSE/Loss', {'train' : loss_val / c}, i)
        
    break


all_pred = []
for j in range(2):
    hi = model.one_step(y, h)
    yi = model.decode(hi)
    all_pred.append(yi)
    y = yi
    h = hi
    
 """    
        
        

        

