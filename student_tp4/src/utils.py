import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)


def fill_na(mat):
    ix,iy = np.where(np.isnan(mat))
    for i,j in zip(ix,iy):
        if np.isnan(mat[i+1,j]):
            mat[i,j]=mat[i-1,j]
        else:
            mat[i,j]=(mat[i-1,j]+mat[i+1,j])/2.
    return mat

def read_temps(path):
    """Lit le fichier de températures"""
    return torch.tensor(fill_na(np.array(pd.read_csv(path).iloc[:,1:])),dtype=torch.float)



class RNN(nn.Module):
    #  TODO:  Implémenter comme décrit dans la question 1
    
    def __init__(self, dim_size, h_size, output_size, F):
        super(RNN, self).__init__()
        self.Wx = torch.nn.Linear(dim_size, h_size)
        self.Wh = torch.nn.Linear(h_size, h_size)
        self.Wo = torch.nn.Linear(h_size, output_size) 
        
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sm = torch.nn.Softmax(dim=1)
        self.sig = torch.nn.Sigmoid()

        self.F = F

        
    def forward(self, x, h0):
        all_h = [h0]
        all_y = []
        for i in range(len(x)):
            hi = self.one_step(x[i], all_h[-1])
            all_h.append(hi)
            all_y.append(self.decode(hi))
            
        return torch.stack(all_h), torch.stack(all_y), self.Wo(all_h[-1])
            

    
    def one_step(self, seq, h):
        return self.tanh( self.Wh(h) + self.Wx(seq) )
        
        
    def decode(self, h):
        y = self.Wo(h)
        return self.F(y)
        
    
#  TODO:  Implémenter les classes Dataset ici
# param nbSeq : nombre de séquences par ville
# param nbVilles : les nb premières villes choisies
def getData(path, task="classif", nbVilles=10, seqLen=200, nbSeq=100):
    
    temp = read_temps(path)[:, :nbVilles]
    #temp = ( temp - temp.min() ).float() / ( temp.max() - temp.min() ).float() 
    #temp = 2*temp - 1
    
    X = []
    y = []
    
    if task == "classif":
        for city in range(temp.shape[1]):
        
            for _ in range(nbSeq):
                    
                randSeq = np.random.randint(0, len(temp) - seqLen)
                
                seq = temp[randSeq:randSeq + seqLen, city].view(1,-1)
                X.append(seq)
                y.append(torch.tensor(city))
            
    
        X = torch.stack(X).permute(2,0,1)
        y = torch.stack(y)
        
        return X, y
    
    elif task == "forecasting":
        for city in range(temp.shape[1]):
            X_i = []
            y_i = []
            for _ in range(nbSeq):
                    
                randSeq = np.random.randint(0, len(temp) - seqLen)
                
                seq = temp[randSeq:randSeq + seqLen, city].view(1,-1)
                X_i.append(seq)
                y_i.append(torch.tensor(city))

            X_i = torch.stack(X_i).permute(2,0,1)
            y_i = torch.stack(y_i)
            
            X.append(X_i)
            y.append(y_i)

    
        return X, y
    
    else:
        return torch.unsqueeze(temp[:nbSeq*nbVilles].T, 2), torch.empty(nbSeq*nbVilles)
    
    
    
    return None





class MonDataset(Dataset):
    
    def __init__(self, X, y):
        self.X = X.double()
        self.y = y.long()
        
    def __getitem__(self, index):
        return self.X[:, index, :], self.y[index]
    
    def __len__(self):
        return len(self.y)
    

