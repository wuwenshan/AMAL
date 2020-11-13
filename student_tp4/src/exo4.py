import string
import unicodedata
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

LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

BATCH_SIZE = 100
SEQ_LEN = 20
HIDDEN_SIZE = 10
OUTPUT_SIZE = len(LETTRES) + 1
DIM_SIZE = len(LETTRES) + 1 

def normalize(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)


def one_hot_encoding(idx):
    x = []
    for i in idx:
        v = torch.zeros(len(LETTRES)+1)
        v[i] = 1
        x.append(v)
    return torch.stack(x)

def getDataTrump(path, seqLen=SEQ_LEN, nbSeq=7400):
    f = open(path, 'r')
    X = []
    c = nbSeq

    
    for line in f:
        for i in range(nbSeq // 74):
 
            rand = np.random.randint(0, len(line) - seqLen)
            seq = string2code( normalize(line[rand:rand + seqLen]) )
            
            if len(seq) == seqLen:
                X.append(one_hot_encoding(seq))
                
            else:
                isLen = False
                
                while not isLen:
                    rand = np.random.randint(0, len(line) - seqLen)
                    seq = string2code( normalize(line[rand:rand + seqLen]) )
                    
                    if len(seq) == seqLen:
                        X.append(one_hot_encoding(seq))
                        isLen = True

                
    X = torch.stack(X).permute(1, 0, 2)

    f.close()
    return X, torch.empty(X.shape[1])



path = "../data/trump_full_speech.txt"

X,y = getDataTrump(path)

print(X.shape, y.shape)

X_test, y_test = getDataTrump(path)

data = DataLoader(MonDataset(X, y), shuffle=True, batch_size=BATCH_SIZE)
data_test = DataLoader(MonDataset(X_test, y_test), shuffle=True, batch_size=BATCH_SIZE)

for x,y in data:
    print(x.shape)
    break



model = RNN(BATCH_SIZE, DIM_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, task="forecasting")
model = model.double()

h0 = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=torch.double)
criterion = torch.nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr = 0.1)

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
    
    for x,y in data:
        model.train()
        with torch.autograd.set_detect_anomaly(False):
            optim.zero_grad()
            c += 1
            x = x.double()
            h, y, all_y = model(x, h0)
            
            loss = criterion(all_y, x)
            loss.backward(retain_graph=True)
            loss_val += loss.item()
            
            optim.step()
            
    """
    for x,_ in data_test:
        model.eval()
        with torch.no_grad():
            c2 += 1
            x = x.permute(1, 0, 2).double()
            half = BATCH_SIZE / 2
            x = x[:, :half, :]
            h, y, all_y = model(x, h0)
    """     
            
    #print("Val : ", loss_val2 / c2)
    loss_train.append(loss_val / c)
    acc_train.append(acc_val / c)
    #loss_test.append(loss_val2 / c2)
    #acc_test.append(acc_val2 / c2)
    #writer.add_scalars('Temp/RNN/Loss/', {'train' : loss_val / c, 'test' : loss_val2/c2}, i)
    #writer.add_scalars('Temp/RNN/Acc/', {'train' : acc_val / c, 'test' : acc_val2/c2}, i)
    writer.add_scalar("Trump/MSELoss/Train", loss_val / c, i)
    



all_pred = []
original = []

for x,_ in data_test:
    model.eval()
    with torch.no_grad():
        c2 += 1
        x = x.double()
        half = int(SEQ_LEN / 2)
        rand = np.random.randint(0, len(x))
        original.append(code2string(torch.max(x[rand], dim=1)[1]))
        
        
        x = x[rand, :half, :]
        
        h, y, all_y = model(x, h0)
        
        for i in range(half):
            h = model.one_step(y, h)
            y = model.decode(h, "classif")
            print(y)
            all_pred.append(code2string(torch.max(y, dim=1)[1]))
        
    break
            
        