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
import torch.distributions.categorical as categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)

LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

BATCH_SIZE = 100
SEQ_LEN = 20
HIDDEN_SIZE = 96
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
    X_emb = []
    c = nbSeq

    for line in f:
        for i in range(nbSeq // 74):
            rand = np.random.randint(0, len(line) - seqLen)
            seq = string2code( normalize(line[rand:rand + seqLen]) )
            
            if len(seq) == seqLen:
                X.append(seq)
                X_emb.append(one_hot_encoding(seq))
                
            else:
                isLen = False
                while not isLen:
                    rand = np.random.randint(0, len(line) - seqLen)
                    seq = string2code( normalize(line[rand:rand + seqLen]) )
                    
                    if len(seq) == seqLen:
                        X.append(seq)
                        X_emb.append(one_hot_encoding(seq))
                        isLen = True

    
    X = torch.stack(X)  
    X_emb = torch.stack(X_emb)

    f.close()
    return X_emb, X



def generation(model, debSentence, lenSentence):
    
    code = string2code(debSentence)
    x = one_hot_encoding(code)
    h = torch.randn(1, HIDDEN_SIZE)
    h = h.to(device)
    sm = torch.nn.Softmax(dim=1)
    output_gen = debSentence
    model = model.double()
    
    for _ in range(lenSentence):
        x = x.to(device)
        all_h, all_y = model(x.double(), h.double())
        proba = sm(all_y[-1])
        nextLet = categorical.Categorical(proba[-1]).sample()
        #nextLet = torch.argmax(proba[-1])
        
        letter = torch.zeros(len(LETTRES)+1)
        letter[nextLet.item()] += 1
        letter = letter.to(device)
        
        x = letter.reshape(1,-1)
        h = all_h[-1]
        output_gen += code2string(torch.tensor([nextLet]))
        
    return output_gen
        
        



path = "../data/trump_full_speech.txt"

X,y = getDataTrump(path)

print(X.shape, y.shape)

X_test, y_test = getDataTrump(path)

data = DataLoader(MonDataset(X, y), shuffle=True, batch_size=BATCH_SIZE)
data_test = DataLoader(MonDataset(X_test, y_test), shuffle=True, batch_size=BATCH_SIZE)


model = RNN(DIM_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, torch.nn.Sigmoid())
model = model.to(device)
model = model.double()

h0 = torch.randn((BATCH_SIZE, HIDDEN_SIZE), dtype=torch.double)
h0 = h0.to(device)
criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr = 0.01)

loss_train = []
loss_test= []
acc_train = []
acc_test = []

writer = SummaryWriter()

aa, bb = next(iter(data))

print(aa.shape, bb.shape)

original = []
generate = []

sm = torch.nn.Softmax(dim=1)

for i in tqdm(range(50)):
    loss_val2 = 0
    loss_val = 0
    acc_val = 0
    acc_val2 = 0
    c = 0
    c2 = 0
    
    for x, target in data:
        model.train()
        with torch.autograd.set_detect_anomaly(False):
            original.append(code2string(target[-1]))
            optim.zero_grad()
            c += 1
            x = x.to(device)
            x = x.double()
            x = x.permute(1, 0, 2)
            target = target.to(device)
            all_h, all_y = model(x, h0)
            
            loss = criterion(all_y.permute(0, 2, 1)[:-1], target.T[1:])
            loss.backward(retain_graph=True)
            loss_val += loss.item()
            
            optim.step()
            
            generate.append(code2string(torch.argmax(sm(all_y.permute(1, 0, 2)[-1]), 1)))
            
    for x, target in data_test:
        with torch.no_grad():
            model.eval()
            c2 += 1
            x = x.to(device)
            x = x.double()
            x = x.permute(1, 0, 2)
            target = target.to(device)
            all_h, all_y = model(x, h0)
            
            loss = criterion(all_y.permute(0, 2, 1)[:-1], target.T[1:])

            loss_val2 += loss.item()            
        
            
    
            
            
    #print("Val : ", loss_val2 / c2)
    loss_train.append(loss_val / c)
    #♦acc_train.append(acc_val / c)
    loss_test.append(loss_val2 / c2)
    #acc_test.append(acc_val2 / c2)
    #writer.add_scalars('Temp/RNN/Loss/', {'train' : loss_val / c, 'test' : loss_val2/c2}, i)
    #writer.add_scalars('Temp/RNN/Acc/', {'train' : acc_val / c, 'test' : acc_val2/c2}, i)
    writer.add_scalar("Trump/MSELoss/Train", loss_val / c, i)
    
    

"""
fig, ax = plt.subplots(figsize=(10,5))
ax.grid()
ax.set(xlabel = "Nombre d'epochs", ylabel="Loss", title=f"Evolution de la loss sur {len(loss_train)} epochs")
plt.plot(np.arange(len(loss_train)), loss_train, label="train")
plt.plot(np.arange(len(loss_test)), loss_test, label="test")
ax.legend()
plt.savefig(f"loss_trump_")  


seq = generation(model, "thank you", 20)

print(seq)
"""