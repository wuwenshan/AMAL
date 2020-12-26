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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)

BATCH_SIZE = 100
HIDDEN_SIZE = 10
OUTPUT_SIZE = 1
    
path = "C:/Users/wuwen/Desktop/EtudeSup/Git/AMAL/student_tp4/data/tempAMAL_train.csv"
path_test = "C:/Users/wuwen/Desktop/EtudeSup/Git/AMAL/student_tp4/data/tempAMAL_test.csv"

X_train, y_train = getData(path, task="forecasting")
X_test, y_test = getData(path_test, task="forecasting")

writer = SummaryWriter()
h0 = torch.randn((BATCH_SIZE, HIDDEN_SIZE), dtype=torch.double)
h0 = h0.to(device)

print("Shape 1 : ", len(X_train), len(y_train), X_train[0].shape, y_train[0].shape)
"""

data_train = DataLoader(MonDataset(X_train, y_train), shuffle=True, batch_size=BATCH_SIZE)
data_test = DataLoader(MonDataset(X_test, y_test), shuffle=True, batch_size=BATCH_SIZE)

model = RNN(1, HIDDEN_SIZE, OUTPUT_SIZE, torch.nn.Sigmoid())
model = model.double()
model = model.to(device)

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
            all_h, all_y = model(x.permute(1,0,2), h0)
            
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
    
    model = RNN(1, HIDDEN_SIZE, OUTPUT_SIZE, torch.nn.Sigmoid())
    model = model.double()
    model = model.to(device)
    
    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr = 0.01)
    
    loss_train = []
    loss_test= []
    acc_train = []
    acc_test = []
    
    expected = []
    predicted = []
    
    for i in tqdm(range(500)):
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
                x = x.double()
                all_h, all_y, _ = model(x.permute(1,0,2), h0)
                
                #print("22 m ", all_h_dec.shape, x.permute(1,0,2).shape)
                #print("22 : ", all_y.shape, x.shape, all_y[1:].shape)
                loss = criterion(all_y[1:], x.permute(1,0,2)[:-1])
                loss.backward()
                loss_val += loss.item()
                
                
                optim.step()
          
        """
        for x, _ in data_test:
            with torch.no_grad():
                model.eval()
                c2 += 1
                pred, dec = model(x.permute(1,0,2), h0)
                
                loss = criterion(dec, x.permute(1,0,2))
                loss_val2 += loss.item()
        """
         
        #print("Val : ", loss_val2 / c2)
        loss_train.append(loss_val / c)
        #loss_test.append(loss_val2 / c2)
        writer.add_scalars('Temp/RNN/MSE/Loss', {'train' : loss_val / c}, i)
        
    break

expected = []
predicted = []

test, y = next(iter(data_test))
test = test.permute(1, 0, 2)
nb = torch.randint(0, len(test), (1, ))
x = test[nb][0]
x = ( x - x.min() ).double() / ( x.max() - x.min() ).double() 
x = 2*x - 1
print(x[0].shape)
expected.append(x.numpy())
x = x[:90]
predicted.append(x.numpy())
h0 = all_h[-1]
x = x.to(device)
print('x shape : ', x.shape)
for i in range(10):
    
    allh, ally, _ = model(x, h0)
    print(allh.shape, ally.shape, x.shape, h0.shape)
    der_h = allh[-1]
    der_y = ally[-1][-1].view(-1,1)
    print("d : ", der_y.shape)
    #x = der_y
    x = torch.cat((x,der_y), 0)
    print("new x :", x.shape)
    h0 = der_h
    #print("2 : ", der_h[-1].shape, der_y[-1].shape)
    predicted.append(x.cpu().detach().numpy())


fig, ax = plt.subplots(figsize=(10,5))
ax.grid()
ax.set(xlabel = "Temps", ylabel="Température normalisée", title=f"Prédiction des 10 derniers pas de temps")
plt.plot(np.arange(len(predicted[-1])), predicted[-1], label="pred")
plt.plot(np.arange(len(expected[0])), expected[0], label="exp")
ax.legend()
plt.savefig(f"pred_exp")  