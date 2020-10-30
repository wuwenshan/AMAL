import torch
from torch.utils.tensorboard import SummaryWriter
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
#import datamaestro
from tqdm import tqdm
from sklearn.datasets import load_boston

writer = SummaryWriter()

# data=datamaestro.prepare_dataset("edu.uci.boston")
# colnames, datax, datay = data.data()
# datax = torch.tensor(datax,dtype=torch.float)
# datay = torch.tensor(datay,dtype=torch.float).reshape(-1,1)

# TODO: 

# X, y = load_boston(return_X_y=True)

# Les données supervisées
x = torch.randn(50, 13)
y = torch.randn(50, 3)

# Les paramètres du modèle à optimiser
w = torch.randn((13, 3), requires_grad=True)
b = torch.randn(3, requires_grad=True)

eps = 0.05

for i in range(100):
    yhat = torch.mm(x, w) + b

    loss = (1/y.shape[0]) * torch.pow(torch.norm(yhat - y), 2)
    loss.backward()
    
    print(f"Itérations {i}: loss {loss.item()}")
    
    w.data -= eps * w.grad.data
    b.data -= eps * b.grad.data
    
    w.grad.data.zero_()
    b.grad.data.zero_()
    