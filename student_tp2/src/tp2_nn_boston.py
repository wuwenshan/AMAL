import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt


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
EPS = 0.01
EPOCHS = 50
BATCH_SIZE = 32
H_SIZE = 128
isNorm = True
grad_descent = "minibatch"

writer = SummaryWriter()

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

"""
model = NN(X.shape[1], H_SIZE, 1)
optimizer = torch.optim.Adam(model.parameters(),lr=EPS)
criterion = torch.nn.MSELoss()
"""

loss_train_tab = []
loss_test_tab = []

model, criterion, optimizer = NN_conteneur(X.shape[1], H_SIZE, 1)

if grad_descent == "batch":
    
    tps = time.time()
    
    for i in range(EPOCHS):
        model.train()
        yhat = model(X_train)
    
        loss_train = criterion(yhat, y_train)
        loss_train.backward()
        
        loss_train_tab.append(loss_train.item())
        #print(f"Itérations {i}: batch/loss/train {loss.item()}")
        #writer.add_scalar('batch/loss/train', loss, i)
        
        optimizer.step()
        optimizer.zero_grad()
        
        with torch.no_grad():
            model.eval()
            yhat = model(X_test)
            loss_test = criterion(yhat, y_test)
            #writer.add_scalar('batch/loss/test', loss_test, i)
            loss_test_tab.append(loss_test.item())
            #print(f"Itérations {i}: batch/loss/test {loss.item()}")
            
        writer.add_scalars('batch/loss', {'train' : loss_train, 'test' : loss_test}, i)
            
    print("Temps : ", time.time() - tps)
            
elif grad_descent == "stochastic":

    
    tps = time.time()
    for i in tqdm(range(EPOCHS)):
        
        cumul_loss = 0
        c = 0
        
        for j in range(X_train.shape[0]-300):
        
            ind = np.random.randint(X_train.shape[0])    
        
            model.train()
            yhat = model(X_train[ind])
            
            loss = criterion(yhat, y_train[ind])
            loss.backward()
            
            cumul_loss += loss.item()
            c += 1
            
            #print(f"Itérations {i}: stoch/loss/train {loss.item()}")
            #writer.add_scalar('stoch/loss/train', loss, i)
            
            optimizer.step()
            optimizer.zero_grad()
            
        loss_train_tab.append(cumul_loss / c)
        with torch.no_grad():
            model.eval()
            yhat = model(X_test)
            loss_test = criterion(yhat, y_test)
            loss_test_tab.append(loss_test.item())
            #writer.add_scalar('stoch/loss/test', loss, i)
            #print(f"Itérations {i}: stoch/loss/test {loss.item()}")
            
        writer.add_scalars('stoch/loss', {'train' : cumul_loss/c, 'test' : loss_test}, i)
            
    print("Temps écoulé : ", time.time() - tps)
    
elif grad_descent == "minibatch":
    
    ind = np.arange(0, X_train.shape[0])
    np.random.shuffle(ind)
    
    X_train = X_train[ind]
    y_train = y_train[ind]
    
    tps = time.time()
    
    for i in range(EPOCHS):
        
        cumul_loss = 0
        c = 0
        
        for j in range(0, len(X_train) - BATCH_SIZE, BATCH_SIZE):
            model.train()
            yhat = model(X_train[j:j+BATCH_SIZE])
        
            loss = criterion(yhat, y_train[j:j+BATCH_SIZE])
            loss.backward()
            
            #print(f"Itérations {i}: minib/loss/train {loss.item()}")
            cumul_loss += loss.item()
            c += 1
            optimizer.step()
            optimizer.zero_grad()
            
        #writer.add_scalar('minib/loss/train', cumul_loss, i)
        loss_train_tab.append(cumul_loss / c)
        with torch.no_grad():
            model.eval()
            yhat = model(X_test)
            loss_test = criterion(yhat, y_test)
            loss_test_tab.append(loss_test)
            #print(f"Itérations {i}: minib/loss/test {loss.item()}")
            #writer.add_scalar('minib/loss/test', loss, i)
            
        writer.add_scalars('minib/loss', {'train' : cumul_loss/c, 'test' : loss_test}, i)
    
    print("Temps : ", time.time() - tps)



if len(loss_train_tab) > 0 and len(loss_test_tab) > 0:
    fig, ax = plt.subplots(figsize=(10,5))
    ax.grid()
    ax.set(xlabel = "Nombre d'epochs", ylabel="Loss", title=f"Evolution de la loss sur {len(loss_train_tab)} epochs")
    plt.plot(np.arange(len(loss_train_tab)), loss_train_tab, label="train")
    plt.plot(np.arange(len(loss_test_tab)), loss_test_tab, label="test")
    ax.legend()
    plt.savefig(f"loss_boston_nn_{grad_descent}")   