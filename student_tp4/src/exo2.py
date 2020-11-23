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

#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)
#torch.cuda.set_device(1)

BATCH_SIZE = 100
    
path = "C:/Users/wuwen/Desktop/EtudeSup/Git/AMAL/student_tp4/data/tempAMAL_train.csv"
path_test = "C:/Users/wuwen/Desktop/EtudeSup/Git/AMAL/student_tp4/data/tempAMAL_test.csv"

X_train, y_train = getData(path)
X_test, y_test = getData(path_test)

print("Shape 1 : ", X_train.shape, y_train.shape)

data_train = DataLoader(MonDataset(X_train,y_train), shuffle=True, batch_size=BATCH_SIZE)
data_test = DataLoader(MonDataset(X_test,y_test), shuffle=True, batch_size=BATCH_SIZE)


model = RNN(1, 20, 10, torch.nn.Softmax(dim=1))
model = model.double()
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr = 0.1)

h0 = torch.randn((BATCH_SIZE, 20),dtype=torch.double)
h0 = h0.to(device)

loss_train = []
loss_test= []
acc_train = []
acc_test = []

writer = SummaryWriter()




for i in tqdm(range(100)):
    loss_val2 = 0
    loss_val = 0
    acc_val = 0
    acc_val2 = 0
    c = 0
    c2 = 0
    
    for x,y in data_train:
        model.train()
        with torch.autograd.set_detect_anomaly(False):
            optim.zero_grad()
            x = ( x - x.min() ).double() / ( x.max() - x.min() ).double() 
            x = 2*x - 1
            c += 1
            x = x.to(device)
            y = y.to(device)
            x = x.double()
            all_h, all_y, last_h = model(x.permute(1,0,2), h0)

            last_y = all_y[-1]
            
            
            cl_pred = torch.max(last_y, dim=1)[1]
            
            acc_val += cl_pred.eq(y).float().mean().item()
            
            
            loss = criterion(last_h, y)
            loss.backward(retain_graph=True)
            loss_val += loss.item()
            
            optim.step()
            
    for x,y in data_test:
        with torch.no_grad():
            x = ( x - x.min() ).double() / ( x.max() - x.min() ).double() 
            x = 2*x - 1
            model.eval()
            x = x.to(device)
            y = y.to(device)
            c2 += 1
            all_h, all_y, last_h = model(x.permute(1,0,2), h0)
            last_y = all_y[-1]
            loss = criterion(last_h, y)
            loss_val2 += loss.item()
            cl_pred = torch.max(last_y, dim=1)[1]
            
            acc_val2 += cl_pred.eq(y).float().mean().item()
            

    loss_train.append(loss_val / c)
    acc_train.append((acc_val / c)*100)
    loss_test.append(loss_val2 / c2)
    acc_test.append((acc_val2 / c2)*100)
    writer.add_scalars('Temp/RNN/Loss/', {'train' : loss_val / c, 'test' : loss_val2/c2}, i)
    writer.add_scalars('Temp/RNN/Acc/', {'train' : acc_val / c, 'test' : acc_val2/c2}, i)
    
