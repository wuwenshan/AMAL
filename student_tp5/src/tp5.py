import csv
import numpy as np
import logging
import time
import string
from itertools import chain
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from textloader import *
from generate import *
import logging
import torch.distributions.categorical as categorical

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DICT_SIZE = 100
EMB_SIZE = 20
BATCH_SIZE = 256
VOCAB_SIZE = 97
H_SIZE = 100

#  TODO:  Implémenter maskedCrossEntropy

def maskedCrossEntropy(output, target):
    """
    padcar = target.clone()
    padcar[padcar>0] = 1
    loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_func(output.permute(0, 2, 1), target)
    #print("Loss : ", loss.shape, padcar.shape)
    loss *= padcar
    
    return torch.sum(loss)    
    """
    #mask = torch.where(target == padcar, torch.tensor(0, device=device), torch.tensor(1, device=device))
    mask = torch.where(target == 0, torch.tensor(0, device=device), torch.tensor(1, device=device))
    #print("ddd ! ", output.shape, target.shape)
    mask = mask.view(mask.shape[0], mask.shape[1], 1)
    mask = mask.to(device)
    output *= mask
    
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output.permute(0, 2, 1), target)
    
    return loss
    


def comparison(output, target):
    sm = torch.nn.Softmax(dim=2)
    pred = torch.max(sm(output), 2)[1]
    
    pred = pred.T
    target = target.T
    
    expected = []
    predicted = []
    
    for i in range(len(pred)):
        expected.append(code2string(target[i]))
        predicted.append(code2string(pred[i]))
        
    return expected, predicted


class RNN(nn.Module):
    #  TODO:  Implémenter comme décrit dans la question 1
    
    def __init__(self, vocab_size, emb_size, h_size, F_os, F_decode):
        super(RNN, self).__init__()
        self.emb = torch.nn.Embedding(vocab_size, emb_size)
        
        self.Wx = torch.nn.Linear(emb_size, h_size)
        self.Wh = torch.nn.Linear(h_size, h_size)
        self.Wo = torch.nn.Linear(h_size, vocab_size) 

        self.F_os = F_os
        self.F_decode = F_decode

        
    def forward(self, x, h0):
        x_emb = self.emb(x.long())
        all_h = [h0]
        all_y = []
        for i in range(len(x_emb)):
            hi = self.one_step(x_emb[i], all_h[-1])
            all_h.append(hi)
            all_y.append(self.decode(hi))
            
        return torch.stack(all_h[1:]), torch.stack(all_y)

    
    def one_step(self, seq, h):
        return self.F_os( self.Wh(h) + self.Wx(seq) )
        
    
    def decode(self, h):
        y = self.Wo(h)
        return y
        #return self.F_decode(y)
    
    
    






class LSTM(torch.nn.Module):
    #  TODO:  Implémenter un LSTM

    def __init__(self, emb_size, h_size, vocab_size):
        super(LSTM, self).__init__()
        self.emb = torch.nn.Embedding(vocab_size, emb_size)
        
        self.Wf = torch.nn.Linear(emb_size + h_size, h_size)
        self.Wi = torch.nn.Linear(emb_size + h_size, h_size)
        self.Wc = torch.nn.Linear(emb_size + h_size, h_size)
        self.Wo = torch.nn.Linear(emb_size + h_size, h_size)
        self.Wout = torch.nn.Linear(h_size, vocab_size)
        
        self.sig = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        
    def forward(self, x, h, c):
        all_h = [h]
        all_c = [c]
        all_y = []
        x_emb = self.emb(x.long())
    
        for i in range(len(x)):
            h, c = self.one_step(x_emb[i], all_h[-1], all_c[-1])
            y = self.decode(h)
    
            all_h.append(h)
            all_c.append(c)
            all_y.append(y)
            
        return torch.stack(all_h[1:]), torch.stack(all_c[1:]), torch.stack(all_y)
    
    def one_step(self, x, h, c):
        h_x = torch.cat((h, x), 1)
        ft = self.sig(self.Wf(h_x))
        it = self.sig(self.Wi(h_x))
        ct = ft * c + it * self.tanh(self.Wc(h_x))
        ot = self.sig(self.Wo(h_x))
        ht = ot * self.tanh(ct)
        
        return ht, ct
    
    def decode(self, h):
        return self.Wout(h)

class GRU(nn.Module):
    #  TODO:  Implémenter un GRU
    
    def __init__(self, emb_size, h_size, vocab_size):
        super(GRU, self).__init__()
        self.emb = torch.nn.Embedding(vocab_size, emb_size)
        
        self.Wz = torch.nn.Linear(emb_size + h_size, h_size, bias=False)
        self.Wr = torch.nn.Linear(emb_size + h_size, h_size, bias=False)
        self.W = torch.nn.Linear(emb_size + h_size, h_size, bias=False)
        self.Wd = torch.nn.Linear(h_size, vocab_size, bias=False)
        
        self.sig = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.sm = torch.nn.Softmax()
        
    def forward(self, x, h):
        all_h = [h]
        all_y = []
        all_emb = []
        all_pred = []
        target_x = []
        
        x_emb = self.emb(x.long()) # (seq_len, batch_size, emb_size)
        
        for i in range(len(x_emb)):

            hi = self.one_step(x_emb[i], all_h[-1])
            yi = self.decode(hi)
            all_h.append(hi)
            all_y.append(yi)
                    
        return torch.stack(all_h[1:]), torch.stack(all_y)#, torch.stack(target_x)#, torch.stack(all_emb), torch.stack(all_pred)
        
    def one_step(self, x, h):
        y = torch.cat((h,x), 1)
        z = self.sig(self.Wz(y))
        r = self.sig(self.Wr(y))
        h = (1 - z) * h + z * self.tanh( self.W( torch.cat((r*h, x), 1) ) )
        
        return h
    
    def decode(self, h):
        return self.Wd(h)
    
    """
    def decode(self, all_h):
        all_y = []
        for i in range(len(all_h)):
            all_y.append(self.Wd(all_h[i]))
        return torch.stack(all_y)
    """
    
    
"""
class GRU(nn.Module):
    #  TODO:  Implémenter un GRU
    def __init__(self, dim_input, dim_output, dim_latent, vocab_size):
        super(GRU, self).__init__()

        self.embedding = nn.Embedding(vocab_size, dim_input)
        self.w_z = nn.Linear(dim_latent+dim_input, dim_latent, bias=False)
        self.w_r = nn.Linear(dim_latent+dim_input, dim_latent, bias=False)
        self.w = nn.Linear(dim_latent+dim_input, dim_latent, bias=False)
        self.w_d = nn.Linear(dim_latent, dim_output)
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_latent = dim_latent
        self.activ_tanh = nn.Tanh()
        self.activ_sig = nn.Sigmoid()
    

    def one_step(self, X, h):
        X_emb = self.embedding(X.long())
        mat_concat = torch.cat((h,X_emb), 1)
        zt = self.activ_sig(self.w_z(mat_concat))
        rt = self.activ_sig(self.w_r(mat_concat))
        ht = (1 - zt) * h + zt * self.activ_tanh(self.w(torch.cat((rt * h, X_emb), 1)))
        return ht


    def forward(self, X, h):
        H = torch.zeros(X.shape[0], X.shape[1], h.shape[1], device=device) # (seq_len, batch, latent)
        h_t_prec = h
        for i in range(len(X)):
            h_t = self.one_step(X[i], h_t_prec)
            h_t_prec = h_t
            H[i] = h_t

        return H


    def decode(self, h):
        D = torch.zeros(h.shape[0], h.shape[1], self.dim_output, device=device)
        for i in range(len(h)):
            D[i] =  self.w_d(h[i])
    
        return D
"""


def entrainement(rnn_type, nb_epoch, lr, device):
    file = open("trump_full_speech.txt", 'r')
    ds = TextDataset(file.read(), maxsent=None, maxlen=50)
    file.close()
    
    loader = DataLoader(ds, collate_fn=collate_fn, batch_size=BATCH_SIZE) 
    
    loss_train = []
    
    if rnn_type == "gru":
        model = GRU(EMB_SIZE, H_SIZE, VOCAB_SIZE)
    
    elif rnn_type == "lstm":
        model = LSTM(EMB_SIZE, H_SIZE, VOCAB_SIZE)
        model = model.to(device)
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        for i in tqdm(range(nb_epoch)):
            v = 0
            c = 0
            for data in loader:
                c += 1
                optim.zero_grad()
                data = data.to(device)
                h0 = torch.randn(data.shape[1], H_SIZE)
                h0 = h0.to(device)
                c0 = torch.randn(data.shape[1], H_SIZE)
                c0 = c0.to(device)
                
                all_h, all_c, all_y = model(data, h0, c0)
                all_y = all_y.to(device)
                
                #print("Here : ", data.shape, all_h.shape, all_c.shape, all_y.shape)
                #break
                pred = all_y[:-1]
                target = data[1:].long()
                loss = maskedCrossEntropy(pred, target.to(device))
                loss.backward(retain_graph=True)
            
                v += loss.item()
                
                optim.step()
            print("Loss : ", v/c)
            loss_train.append(v/c)
            
        fig, ax = plt.subplots(figsize=(10,5))
        ax.grid()
        ax.set(xlabel = "Nombre d'epochs", ylabel="Loss", title=f"Evolution de la loss sur {len(loss_train)} epochs")
        plt.plot(np.arange(len(loss_train)), loss_train, label="train")
        ax.legend()
        plt.savefig(f"loss_train_gen_")  

            
            
        return model
        
    else:
        print("Unknown model : ", rnn_type)
        return None
        
    return None

    
    

#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot



"""
file = open("trump_full_speech.txt", 'r')
ds = TextDataset(file.read(), maxlen=None)
file.close()

loader = DataLoader(ds, collate_fn=collate_fn, batch_size=BATCH_SIZE)

#model = RNN(VOCAB_SIZE, EMB_SIZE, H_SIZE, torch.nn.ReLU(), torch.nn.Tanh())
model = LSTM(EMB_SIZE, H_SIZE, VOCAB_SIZE)
#model = GRU(EMB_SIZE, VOCAB_SIZE, H_SIZE, VOCAB_SIZE)
#model = model.to(device)

h0 = torch.randn(BATCH_SIZE, H_SIZE)
h0 = h0.to(device)

criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.1)
writer = SummaryWriter()

sm = torch.nn.Softmax(dim=1)

loss_val = []

all_expected = []
all_predicted = []

test_expec = []
test_pred = []

test_val = []
tab = []

liste_h = []
liste_h.append(h0)




for i in tqdm(range(20)):
    v = 0
    c = 0
    v2 = 0
    c2 = 0
    for data in loader:
        c += 1
        optim.zero_grad()
        data = data.to(device)
        h0 = torch.randn(data.shape[1], H_SIZE)
        h0 = h0.to(device)
        c0 = torch.randn(data.shape[1], H_SIZE)
        c0 = c0.to(device)
        #print("mdr : ", data.shape, h0.shape)
        
        batch = data
        batch = batch.to(device)
        H, pred = model(batch, h0, c0)
        H = H.to(device)
        pred = pred.to(device)
        #print("H shape : ", H.shape)
        #pred = model.decode(H)
        #print("pred shape : ", pred.shape)
        
        pred = pred.permute(0,2,1)
        real = batch[1:].long()
        pad = real.clone()
        pad[pad>0] = 1
        #print("narrow : ", pred.narrow(0,0,len(pred)-1).shape, real.shape, pad.shape)
        #loss = maskedCrossEntropy(pred.narrow(0,0,len(pred)-1), real.to(device), pad.to(device))
        loss = maskedCrossEntropy(pred[:-1], real.to(device), pad.to(device))
        #output = maskedCrossEntropy(pred[:-1], real, )
        loss.backward(retain_graph=True)
        
        #target = torch.zeros_like(data)
        #target[:-1] = data[1:]
        #target[-1] = data[0]
        
        
        
        all_h = model(data, h0) # data = (seq_len, batch_size)
        all_y = model.decode(all_h)
        pred = all_y.permute(0,2,1)
        target = data[1:].long()
        pad = target.clone()
        pad[pad>0] = 1
        
        #print("all shapes : ", h.shape, y.shape, target.shape, data.shape)
        #break
        
        # h shape (183, 50, 10) (seq_len, batch_size, h_size)
        # y shape (183, 50, 97) (seq_len, batch_size, vocab_size)
        #print("Shape \n")
        #print("ola : ", h.shape, y.shape, data.shape)
        #y = y.to(device)
        loss = maskedCrossEntropy(pred.narrow(0,0,len(pred)-1), target, pad)

        #expected, predicted = comparison(y[:-1], target)
        #all_expected.append(expected)
        #all_predicted.append(predicted)
        

        #☻loss = criterion(pred[:-1], target.long())
        v += loss.item()
        

        #•last_h = h[-1].detach()

        #liste_h.append(last_h)

        #loss.backward()
        
        optim.step()
    print("loss : ", v/c)
    loss_val.append(v/c)
    #writer.add_scalar("Trump/CEL", v/c, i)
    #break
    
"""          
model = entrainement("lstm", 20, 0.1, device)   
     
seq = generate(model, "The world is", device)

all_seq = generate_beam(model, 10, start="The world is", maxlen=50, device=device)

print('Sequence générée par beam search\n', seq)
#♥seq, _ = generate(model, None, None, 1, start="The world is", maxlen=200)
#print('Sequence générée par sampling\n : The world is ', seq)

for i in range(all_seq[-1].shape[1]):
    print(code2string(all_seq[-1][:,i]))

