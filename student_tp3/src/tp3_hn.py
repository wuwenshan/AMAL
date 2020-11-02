from pathlib import Path
import os
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# Téléchargement des données

from datamaestro import prepare_dataset
ds = prepare_dataset("com.lecun.mnist");
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()

print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape, type(train_images))

# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Pour visualiser
# Les images doivent etre en format Channel (3) x Hauteur x Largeur
images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1,3,1,1).double()/255.
# Permet de fabriquer une grille d'images
images = make_grid(images)
# Affichage avec tensorboard
writer.add_image(f'samples', images, 0)


savepath = Path("model.pch")

#  TODO: 
    
## Gérer les données avec Dataset et Dataloader

class MonDataset(Dataset):

	def __init__(self, x, y):
		self.x = torch.from_numpy(x).reshape(-1, x.shape[1] * x.shape[2]) / 255
		self.y = torch.from_numpy(y)
		self.x = self.x.double()
		self.y = self.y.double()
	
	def __getitem__(self, index):
		return self.x[index].double(), self.y[index].double()

	def __len__(self):
		return len(self.y)


train_data = DataLoader(MonDataset(train_images, train_labels), shuffle=True, batch_size=128)
test_data = DataLoader(MonDataset(test_images, test_labels), shuffle=True, batch_size=128)


class MonHighwayNetwork(torch.nn.Module):
    
    def __init__(self, x_size):
        super(MonHighwayNetwork, self).__init__()
        #self.H = torch.nn.ModuleList([torch.nn.Linear(x_size, x_size) for _ in range(10)])
        #self.T = torch.nn.ModuleList([torch.nn.Linear(x_size, x_size) for _ in range(10)])

        self.H = torch.nn.Linear(x_size, x_size)
        self.T = torch.nn.Linear(x_size, x_size)
        self.Sig = torch.nn.Sigmoid()
        self.Rel = torch.nn.ReLU()

        
    def forward(self, x):
        
        #for layer in range(10):
            #gate = self.Sig(self.T[layer](x))
            #y = self.Rel(self.H[layer](x)) * gate + (1 - gate) * x
        gate = self.Sig(self.T(x))
        y = self.Rel(self.H(x)) * gate + x * (1 - gate)
        return y
        



class State:
	def __init__(self, model, optim):
		self.model = model
		self.optim = optim
		self.epoch, self.iteration = 0, 0
        
    
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
if savepath.is_file():
	with savepath.open("rb") as fp:
		state = torch.load(fp)
else:
"""
	
model = MonHighwayNetwork(train_images.shape[1] * train_images.shape[2])
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = torch.nn.CrossEntropyLoss()
state = State(model, optimizer)

loss_train_tab = []
loss_test_tab = []

for i in tqdm(range(50)):
    loss_train = 0
    loss_test = 0
    c1 = 0
    c2 = 0

    for data,label in train_data:
        model.train()
        state.optim.zero_grad()
        
        c1 += 1
        data = data.to(device)
        label = label.to(device)
        pred = model(data.float())
        
        loss = criterion(pred, label.long())
        loss.backward()
        
        loss_train += loss.item()
        
        state.optim.step()
		
        state.iteration += 1

	
    with savepath.open("wb") as fp:
		
        state.epoch = i + 1
		
        torch.save(state, fp)

    loss_train_tab.append(loss_train / c1)
    for data,label in test_data:
        
        with torch.no_grad():
            model.eval()
    		
            c2 += 1
    		
            data = data.to(device)
            label = label.to(device)	
            pred = model(data.float())
            
            loss = criterion(pred, label.long())
            loss_test += loss.item()
            
    loss_test_tab.append(loss_test / c2)

    writer.add_scalars('MNIST/AE/Loss', {"loss_train" : loss_train/c1, "loss_test" : loss_test/c2}, i)



if len(loss_train_tab) > 0 and len(loss_test_tab) > 0:
    fig, ax = plt.subplots(figsize=(10,5))
    ax.grid()
    ax.set(xlabel = "Nombre d'epochs", ylabel="Loss", title=f"Evolution de la loss sur {len(loss_train_tab)} epochs")
    plt.plot(np.arange(len(loss_train_tab)), loss_train_tab, label="train")
    plt.plot(np.arange(len(loss_test_tab)), loss_test_tab, label="test")
    ax.legend()
    plt.savefig(f"loss_mnist_hn_")  




	
		
