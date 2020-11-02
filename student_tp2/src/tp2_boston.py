import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
import math
import matplotlib.pyplot as plt
# import datamaestro

TEST_SIZE = 0.2
EPS = 0.01
EPOCHS = 20    
BATCH_SIZE = 20
isNorm = True
grad_descent = "minibatch"

writer = SummaryWriter()

# data=datamaestro.prepare_dataset("edu.uci.boston")
# colnames, datax, datay = data.data()
# datax = torch.tensor(datax,dtype=torch.float)
# datay = torch.tensor(datay,dtype=torch.float).reshape(-1,1)

X, y = load_boston(return_X_y=True)

y = y.reshape(-1,1)


if isNorm:
    X = ( X - np.mean(X, 0) ) / np.std(X, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Les paramètres du modèle à optimiser
w = torch.normal(mean=0, std=1/math.sqrt(X_train.shape[0]),size=(X_train.shape[1], y_train.shape[1]), requires_grad=True)
b = torch.randn(1, requires_grad=True)

loss_train_tab = []
loss_test_tab = []

if grad_descent == "batch":

    for i in tqdm(range(EPOCHS)):
        yhat = torch.mm(X_train, w) + b
    
        loss_train = (1/y_train.shape[0]) * torch.pow(torch.norm(yhat - y_train), 2)
        loss_train.backward()
        
        loss_train_tab.append(loss_train)
        #print(f"Itérations {i}: batch/loss/train {loss.item()}")
        #writer.add_scalar('batch/loss/train', loss_train, i)
        
        w.data -= EPS * w.grad
        b.data -= EPS * b.grad
        
        w.grad.data.zero_()
        b.grad.data.zero_()
        
        with torch.no_grad():
            yhat = torch.mm(X_test, w) + b
            loss_test = (1/y_test.shape[0]) * torch.pow(torch.norm(yhat - y_test), 2)
            loss_test_tab.append(loss_test)
            #writer.add_scalar('batch/loss/test', loss, i)
            #print(f"Itérations {i}: batch/loss/test {loss.item()}")
            
        writer.add_scalars('batch/loss', {'loss train' : loss_train, 'loss test' : loss_test}, i)
            
elif grad_descent == "stochastic":
    
    for i in tqdm(range(EPOCHS)):
        
        cumul_loss = 0
        t = 0
        
        for _ in range(X_train.shape[0]):
            
            j = np.random.randint(X_train.shape[0])
            yhat = torch.mm(X_train[j].reshape(1,-1), w) + b
            
            loss_train = (1/y_train[j].shape[0]) * torch.pow(torch.norm(yhat - y_train[j].reshape(1,-1)), 2)
            loss_train.backward()
            
            #print(f"Itérations {i}: stoch/loss/train {loss_train.item()}")
            #writer.add_scalar('Test/stoch/loss/train', loss_train, i)
            
            cumul_loss += loss_train.item()
            
            w.data -= EPS * w.grad.data
            b.data -= EPS * b.grad.data
            
            
            w.grad.data.zero_()
            b.grad.data.zero_()
        
            t += 1
            
        loss_train_tab.append(cumul_loss / t)
        
        with torch.no_grad():
            yhat = torch.mm(X_test, w) + b
            loss_test = (1/y_test.shape[0]) * torch.pow(torch.norm(yhat - y_test), 2)
            loss_test_tab.append(loss_test)
            #writer.add_scalar('stoch/loss/test', loss, i)
            #print(f"Itérations {i}: stoch/loss/test {loss_test.item()}")
    
        writer.add_scalars('stoch/loss', {'loss train' : cumul_loss / t, 'loss test' : loss_test}, i)
    
elif grad_descent == "minibatch":
    
    ind = np.arange(0, X_train.shape[0])
    np.random.shuffle(ind)
    
    X_train = X_train[ind]
    y_train = y_train[ind]
    
    for i in range(EPOCHS):
        cumul_loss = 0
        c = 0
        for j in range(0, len(X_train) - BATCH_SIZE, BATCH_SIZE):
            c += 1
            yhat = torch.mm(X_train[j:j+BATCH_SIZE], w) + b
        
            loss = (1/y_train[j:j+BATCH_SIZE].shape[0]) * torch.pow(torch.norm(yhat - y_train[j:j+BATCH_SIZE]), 2)
            loss.backward()
            
            #print(f"Itérations {i}: minib/loss/train {loss.item()}")
            cumul_loss += loss.item()
            
            w.data -= EPS * w.grad.data
            b.data -= EPS * b.grad.data
            
            w.grad.data.zero_()
            b.grad.data.zero_()
            
        #writer.add_scalar('minib/loss/train', cumul_loss, i)
            
        loss_train_tab.append(cumul_loss / c)
        with torch.no_grad():
            yhat = torch.mm(X_test, w) + b
            loss = (1/y_test.shape[0]) * torch.pow(torch.norm(yhat - y_test), 2)
            loss_test_tab.append(loss)
            #print(f"Itérations {i}: minib/loss/test {loss.item()}")
            #writer.add_scalar('minib/loss/test', loss, i)
            
        writer.add_scalars('minib/loss', {'loss_train' : cumul_loss/c, 'loss_test' : loss}, i)
        
        
if len(loss_train_tab) > 0 and len(loss_test_tab) > 0:
    fig, ax = plt.subplots(figsize=(10,5))
    ax.grid()
    ax.set(xlabel = "Nombre d'epochs", ylabel="Loss", title=f"Evolution de la loss sur {len(loss_train_tab)} epochs")
    plt.plot(np.arange(len(loss_train_tab)), loss_train_tab, label="train")
    plt.plot(np.arange(len(loss_test_tab)), loss_test_tab, label="test")
    ax.legend()
    plt.savefig(f"loss_boston_{grad_descent}")   