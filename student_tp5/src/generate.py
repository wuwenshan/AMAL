from textloader import *
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.distributions.categorical as categorical
import torch.nn.functional as F
import numpy as np
import math

#  TODO:  Ce fichier contient les différentes fonction de génération

def generate(rnn, start, device):
    #  TODO:  Implémentez la génération à partir du RNN, et d'une fonction decoder qui renvoie les logits (logarithme de probabilité à une constante près, i.e. ce qui vient avant le softmax) des différentes sorties possibles

    x = string2code(start)
    x = x.to(device)
    h0 = torch.randn(1, 100)
    h0 = h0.to(device)
    c0 = torch.randn(1, 100)
    c0 = c0.to(device)
    seq = start
    
    sm = torch.nn.Softmax(dim=1)
    
    
    x = x.view(-1, 1)
    for i in range(50):
        rnn.eval()
        all_h, all_c, all_y = rnn(x, h0, c0)
        x_sm = sm(all_y[-1])
        #pred = categorical.Categorical(x_sm).sample()
        pred = torch.argmax(x_sm[-1])
        pred = pred.view(-1,1)
        x = torch.cat((x, pred), 0)
        h0 = all_h[-1]
        
        
        
        seq = seq + code2string(pred[0])
    
    return seq








def generate_beam(rnn, k, device, start="", maxlen=200):
    #  TODO:  Implémentez le beam Search
    all_seq_codes = []
    
    x = string2code(start)
    x = x.to(device)
    
    sm = torch.nn.Softmax(dim=2)
    all_seq_codes.append(x.view(-1,1))
    
    isStarting = True
    
    while (len(all_seq_codes[-1]) <= maxlen):
        x = all_seq_codes[-1]
        
        h0 = torch.randn(x.shape[1], 100)
        h0 = h0.to(device)
        c0 = torch.randn(x.shape[1], 100)
        c0 = c0.to(device)
        
        rnn.eval()
        all_h, all_c, all_y = rnn(x, h0, c0)
        value, pred = torch.topk(sm(all_y), k)
        h0 = all_h[-1]
        c0 = all_c[-1]
            
        if isStarting:
            x = torch.cat(k*[x], 1)
            new_x = torch.cat((x, pred[-1]), 0)
            all_seq_codes.append(new_x)
            isStarting = False
        
        else:
            indices = torch.cat(k*[torch.arange(k).view(-1, 1)], 1)
            indices = indices.view(1, k*k)
            values = value[-1].view(1, k*k)
            
            val,ind = torch.sort(values, descending=True)
            
            ind = ind[0][:k]
            code = pred[-1].view(1, k*k)
            code = code[0][ind]
            seq = indices[0][ind]
            all_seq = []
            
            for i in range(len(seq)):
                new_x = torch.cat((x[:,seq[i]].view(-1,1), code[i].view(-1,1)), 0)
                all_seq.append(new_x)
            all_seq_codes.append(torch.squeeze(torch.stack(all_seq, 1), 2))
   
    
    return all_seq_codes
     

# p_nucleus
def p_nucleus(decoder, k: int):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        decoder: renvoie les logits étant donné l'état du RNN
        k (int): [description]
    """
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
            h (torch.Tensor): L'état à décoder
        """
        #  TODO:  Implémentez le Nucleus sampling ici (pour un état s)
    return compute
