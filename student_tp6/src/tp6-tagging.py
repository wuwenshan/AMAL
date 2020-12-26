import itertools
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from datamaestro import prepare_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import List
import time
logging.basicConfig(level=logging.INFO)

ds = prepare_dataset('org.universaldependencies.french.gsd')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Format de sortie décrit dans
# https://pypi.org/project/conllu/

class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        self.oov =  oov
        self.id2word = [ "PAD"]
        self.word2id = { "PAD" : Vocabulary.PAD}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self,idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self,idx: List[int]):
        return [self.getword(i) for i in idx]



class TaggingDataset():
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        self.sentences = []

        for s in data:
            self.sentences.append(([words.get(token["form"], adding) for token in s], [tags.get(token["upostag"], adding) for token in s]))
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, ix):
        return self.sentences[ix]


def collate(batch):
    """Collate using pad_sequence"""
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    return tuple(pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2))


logging.info("Loading datasets...")
words = Vocabulary(True)
tags = Vocabulary(False)
train_data = TaggingDataset(ds.train, words, tags, True)
dev_data = TaggingDataset(ds.validation, words, tags, True)
test_data = TaggingDataset(ds.test, words, tags, False)


logging.info("Vocabulary size: %d", len(words))


BATCH_SIZE = 100
EMB_SIZE = 30
H_SIZE = 60
NB_EPOCH = 50

train_loader = DataLoader(train_data, collate_fn=collate, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_data, collate_fn=collate, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, collate_fn=collate, batch_size=BATCH_SIZE)


#  TODO:  Implémentez le modèle et la boucle d'apprentissage


class TaggingModel(torch.nn.Module):

	def __init__(self, vocab_size, emb_size, h_size, tag_size):
		super(TaggingModel, self).__init__()
		self.h_size = h_size		

		self.embedding = torch.nn.Embedding(vocab_size, emb_size, padding_idx=0)
		self.lstm = torch.nn.LSTM(emb_size, h_size)
		self.linear = torch.nn.Linear(h_size, tag_size)

	def forward(self, x):
		emb = self.embedding(x)
		batch_size = x.shape[1]
		h0 = torch.randn(1, batch_size, self.h_size)
		h0 = h0.to(device)
		c0 = torch.randn(1, batch_size, self.h_size)
		c0 = c0.to(device)
		output, (hn, cn) = self.lstm(emb, (h0, c0))
		return self.linear(output)


model = TaggingModel(words.__len__(), EMB_SIZE, H_SIZE, tags.__len__())
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)
sm = torch.nn.Softmax(dim=2)

loss_train = []
loss_test = []
acc_train = []
acc_test = []

infos = []

for i in tqdm(range(NB_EPOCH)):
    loss_val = 0
    acc_val = 0
    c = 0
    for x,y in train_loader:
        c += 1
		#print("\nx, y : ", x.shape, y.shape)
        optim.zero_grad()
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
		#print('pred shape : ', pred.shape)
        loss = criterion(pred.permute(0, 2, 1), y)
        loss_val += loss.item()
        cl = torch.max(sm(pred),2)[1]
        acc_val += cl.eq(y).float().mean()

        loss.backward()
        optim.step()

    loss_train.append(loss_val / c)
    acc_train.append((acc_val/c)*100)
    acc_val = 0
    loss_val = 0
    c = 0
    
    for x,y in test_loader:
        with torch.no_grad():
            c += 1
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred.permute(0, 2, 1), y)
            loss_val += loss.item()
            cl = torch.max(sm(pred),2)[1]
            acc_val += cl.eq(y).float().mean()

    acc_test.append((acc_val/c)*100)
    loss_test.append(loss_val / c)

    idx = torch.randint(0, y.shape[1], (1,))
    phrase = words.getwords(x[:, idx])
    original = tags.getwords(y[:,idx])
    pred = tags.getwords(cl[:,idx])

    infos.append((phrase, original, pred))

fig, ax = plt.subplots(figsize=(10,5))
ax.grid()
ax.set(xlabel = "Nombre d'epochs", ylabel="Loss", title=f"Evolution de la loss sur {len(loss_train)} epochs")
plt.plot(np.arange(len(loss_train)), loss_train, label="train")
plt.plot(np.arange(len(loss_test)), loss_test, label="test")
ax.legend()
plt.savefig(f"loss_tagging_")  

fig, ax = plt.subplots(figsize=(10,5))
ax.grid()
ax.set(xlabel = "Nombre d'epochs", ylabel="Accuracy (%)", title=f"Evolution de l'accuracy sur {len(acc_train)} epochs")
plt.plot(np.arange(len(acc_train)), acc_train, label="train")
plt.plot(np.arange(len(acc_test)), acc_test, label="test")
ax.legend()
plt.savefig(f"acc_tagging_")  

p1, o1, pred1 = infos[1]

p2, o2, pred2 = infos[25]

p3, o3, pred3 = infos[NB_EPOCH-1]

print("Acc train : ", acc_train)
print("Acc test : ", acc_test)

print("Phrase 1 : ", p1)
print("Tag 1 : ", o1)
print("Pred 1 : ", pred1)

print("Phrase 2 : ", p2)
print("Tag 2 : ", o2)
print("Pred 2 : ", pred2)

print("Phrase 3 : ", p3)
print("Tag 3 : ", o3)
print("Pred 3 : ", pred3)
