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

loss_train_tab = []
loss_test_tab = []

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

## Implémentation d'un autoencodeur

class MonAutoEncoder(torch.nn.Module):

	def __init__(self, x_size, h_size):
		super(MonAutoEncoder, self).__init__()
		self.L1 = torch.nn.Linear(x_size, h_size)
		self.L2 = torch.nn.Linear(h_size, x_size)
		self.F1 = torch.nn.ReLU()
		self.F2 = torch.nn.Sigmoid()


	def forward(self, x):
		encoder = self.F1(self.L1(x))
		self.L2.weight = torch.nn.Parameter(self.L1.weight.T)
		decoder = self.F2(self.L2(encoder))
		return decoder



class State:
	def __init__(self, model, optim):
		self.model = model
		self.optim = optim
		self.epoch, self.iteration = 0, 0
        
        
def comparison(pred, real):
    pred = np.array(pred.cpu())
    real = np.array(real.cpu())
    img_ind = np.random.randint(0, len(pred))
    _, axarr = plt.subplots(2,sharex=True,sharey=True)
    axarr[0].set_title('Image originale')
    axarr[0].imshow(real[img_ind].reshape(28,28))
    axarr[1].set_title('Image reconstruite')
    axarr[1].imshow(pred[img_ind].reshape(28,28))
    plt.savefig("figHN"+str(img_ind))
    
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
if savepath.is_file():
	with savepath.open("rb") as fp:
		state = torch.load(fp)
else:
"""
	
model = MonAutoEncoder(train_images.shape[1] * train_images.shape[2], 128)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()
state = State(model, optimizer)


for i in tqdm(range(50)):
    loss_train = 0
    loss_test = 0
    c1 = 0
    c2 = 0

    for data,label in train_data:
        model.train()
        state.optim.zero_grad()
        c1+= 1
        data = data.to(device)

        pred = model(data.float())
        loss = criterion(pred, data.float())	
        loss.backward()
        
        loss_train += loss.item()
        
        state.optim.step()
		
        state.iteration += 1
		
    loss_train_tab.append(loss_train / c1)
    #writer.add_scalar('MNIST/HN/Loss', loss_train/c1, i)
	
    with savepath.open("wb") as fp:
		
        state.epoch = i + 1
		
        torch.save(state, fp)

    
    for data,label in test_data:
        
        with torch.no_grad():
            model.eval()
    		
            c2 += 1
    		
            data = data.to(device)
            pred = model(data.float())
            
            loss = criterion(pred, data.float())
            loss_test += loss.item()
            
    loss_test_tab.append(loss_test / c2)


    writer.add_scalars('MNIST/AE/Loss', {"loss_train" : loss_train/c1, "loss_test" : loss_test/c2}, i)

    
    if i >= 197:
        comparison(pred, data)
        

	

if len(loss_train_tab) > 0 and len(loss_test_tab) > 0:
    fig, ax = plt.subplots(figsize=(10,5))
    ax.grid()
    ax.set(xlabel = "Nombre d'epochs", ylabel="Loss", title=f"Evolution de la loss sur {len(loss_train_tab)} epochs")
    plt.plot(np.arange(len(loss_train_tab)), loss_train_tab, label="train")
    plt.plot(np.arange(len(loss_test_tab)), loss_test_tab, label="test")
    ax.legend()
    plt.savefig(f"loss_mnist_ae_")   


	
		
