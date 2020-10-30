import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np

TEST_SIZE = 0.2
EPS = 1e-3
EPOCHS = 2000
BATCH_SIZE = 32
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

# Les paramètres du modèle à optimiser
w = torch.randn((X_train.shape[1], 1), requires_grad=True)
b = torch.randn(1, requires_grad=True)

if grad_descent == "batch":

    for i in range(EPOCHS):
        yhat = torch.mm(X_train, w) + b
    
        loss = (1/y_train.shape[0]) * torch.pow(torch.norm(yhat - y_train), 2)
        loss.backward()
        
        print(f"Itérations {i}: batch/loss/train {loss.item()}")
        writer.add_scalar('batch/loss/train', loss, i)
        
        w.data -= EPS * w.grad.data
        b.data -= EPS * b.grad.data
        
        w.grad.data.zero_()
        b.grad.data.zero_()
        
        with torch.no_grad():
            yhat = torch.mm(X_test, w) + b
            loss = (1/y_test.shape[0]) * torch.pow(torch.norm(yhat - y_test), 2)
            writer.add_scalar('batch/loss/test', loss, i)
            print(f"Itérations {i}: batch/loss/test {loss.item()}")
            
elif grad_descent == "stochastic":
    idx = torch.randperm(len(X_train))
    t = 0
    
    for i in range(EPOCHS):
        yhat = torch.mm(X_train[idx[t]].reshape(1,-1), w) + b
        
        loss = (1/y_train[idx[t]].shape[0]) * torch.pow(torch.norm(yhat - y_train[idx[t]].reshape(1,-1)), 2)
        loss.backward()
        
        print(f"Itérations {i}: stoch/loss/train {loss.item()}")
        writer.add_scalar('stoch/loss/train', loss, i)
        
        w.data -= EPS * w.grad.data
        b.data -= EPS * b.grad.data
        
        w.grad.data.zero_()
        b.grad.data.zero_()
        
        t += 1
        
        with torch.no_grad():
            yhat = torch.mm(X_test, w) + b
            loss = (1/y_test.shape[0]) * torch.pow(torch.norm(yhat - y_test), 2)
            writer.add_scalar('stoch/loss/test', loss, i)
            print(f"Itérations {i}: stoch/loss/test {loss.item()}")
    
elif grad_descent == "minibatch":
    
    for i in range(EPOCHS):
        cumul_loss = 0
        for j in range(0, len(X_train) - BATCH_SIZE, BATCH_SIZE):
        
            yhat = torch.mm(X_train[j:j+BATCH_SIZE], w) + b
        
            loss = (1/y_train[j:j+BATCH_SIZE].shape[0]) * torch.pow(torch.norm(yhat - y_train[j:j+BATCH_SIZE]), 2)
            loss.backward()
            
            print(f"Itérations {i}: minib/loss/train {loss.item()}")
            cumul_loss += loss.item()
            
            w.data -= EPS * w.grad.data
            b.data -= EPS * b.grad.data
            
            w.grad.data.zero_()
            b.grad.data.zero_()
            
        writer.add_scalar('minib/loss/train', cumul_loss, i)
            
        with torch.no_grad():
            yhat = torch.mm(X_test, w) + b
            loss = (1/y_test.shape[0]) * torch.pow(torch.norm(yhat - y_test), 2)
            print(f"Itérations {i}: minib/loss/test {loss.item()}")
            writer.add_scalar('minib/loss/test', loss, i)