import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np


class NN(torch.nn.Module):
    
    def __init__(self, x_size, h_size, out_size):
        super(NN, self).__init__()
        self.L1 = torch.nn.Linear(x_size, h_size)
        self.L2 = torch.nn.Linear(h_size, out_size)
        self.Tanh = torch.nn.Tanh()
        
    def forward(self, x):
        y = self.L2( self.Tanh( self.L1(x) ) )
        return y
    
def NN_conteneur(x_size, h_size, out_size):
    model = torch.nn.Sequential(
            torch.nn.Linear(x_size, h_size),
            torch.nn.Tanh(),
            torch.nn.Linear(h_size, out_size),
            )
    criterion = torch.nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr=EPS)
    return model, criterion, optim
    
TEST_SIZE = 0.2
EPS = 1e-2
EPOCHS = 100
BATCH_SIZE = 32
H_SIZE = 128
isNorm = True
grad_descent = "batch"

writer = SummaryWriter()

X, y = load_boston(return_X_y=True)

y = y.reshape(-1,1)

if isNorm:
    X = ( X - np.mean(X, 0) ) / np.var(X, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

"""
model = NN(X.shape[1], H_SIZE, 1)
optimizer = torch.optim.Adam(model.parameters(),lr=EPS)
criterion = torch.nn.MSELoss()
"""

model, criterion, optimizer = NN_conteneur(X.shape[1], H_SIZE, 1)

if grad_descent == "batch":

    for i in range(EPOCHS):
        model.train()
        yhat = model(X_train)
    
        loss = criterion(yhat, y_train)
        loss.backward()
        
        print(f"Itérations {i}: batch/loss/train {loss.item()}")
        writer.add_scalar('batch/loss/train', loss, i)
        
        optimizer.step()
        optimizer.zero_grad()
        
        with torch.no_grad():
            model.eval()
            yhat = model(X_test)
            loss = criterion(yhat, y_test)
            writer.add_scalar('batch/loss/test', loss, i)
            print(f"Itérations {i}: batch/loss/test {loss.item()}")
            
elif grad_descent == "stochastic":
    idx = torch.randperm(len(X_train))
    t = 0
    
    for i in range(EPOCHS):
        model.train()
        yhat = model(X_train[idx[t]])
        
        loss = criterion(yhat, y_train[idx[t]])
        loss.backward()
        
        print(f"Itérations {i}: stoch/loss/train {loss.item()}")
        writer.add_scalar('stoch/loss/train', loss, i)
        
        optimizer.step()
        optimizer.zero_grad()
        
        t += 1
        
        with torch.no_grad():
            model.eval()
            yhat = model(X_test)
            loss = criterion(yhat, y_test)
            writer.add_scalar('stoch/loss/test', loss, i)
            print(f"Itérations {i}: stoch/loss/test {loss.item()}")
    
elif grad_descent == "minibatch":
    
    for i in range(EPOCHS):
        cumul_loss = 0
        for j in range(0, len(X_train) - BATCH_SIZE, BATCH_SIZE):
            model.train()
            yhat = model(X_train[j:j+BATCH_SIZE])
        
            loss = criterion(yhat, y_train[j:j+BATCH_SIZE])
            loss.backward()
            
            print(f"Itérations {i}: minib/loss/train {loss.item()}")
            cumul_loss += loss.item()
            
            optimizer.step()
            optimizer.zero_grad()
            
        writer.add_scalar('minib/loss/train', cumul_loss, i)
            
        with torch.no_grad():
            model.eval()
            yhat = model(X_test)
            loss = criterion(yhat, y_test)
            print(f"Itérations {i}: minib/loss/test {loss.item()}")
            writer.add_scalar('minib/loss/test', loss, i)
    
    

