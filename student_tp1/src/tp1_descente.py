import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context
import numpy as np
import matplotlib.pyplot as plt

# Les données supervisées
x = torch.randn(50, 13)
y = torch.randn(50, 3)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3)
b = torch.randn(3)

epsilon = 0.01

ctx = Context() 
ctx2 = Context()

linear = Linear()
mse = MSE()

loss_tab = []

writer = SummaryWriter()
for n_iter in range(200):
    ##  TODO:  Calcul du forward (loss)
    yhat = linear.forward(ctx, x, w, b)
    loss = mse.forward(ctx2, yhat, y)

    # `loss` doit correspondre au coût MSE calculé à cette itération
    # on peut visualiser avec
    # tensorboard --logdir runs/
    # writer.add_scalar('Loss', loss, n_iter)
    # Sortie directe
    if n_iter % 10 == 0:
        print(f"Itérations {n_iter}: loss {loss}")
    loss_tab.append(loss)

    ##  TODO:  Calcul du backward (grad_w, grad_b)
    d_yhat, d_y = mse.backward(ctx2, 1)
    _, grad_w, grad_b = linear.backward(ctx, d_yhat)

    ##  TODO:  Mise à jour des paramètres du modèle
    w = w - epsilon*grad_w
    b = b - epsilon*grad_b
    

if len(loss_tab) > 0:

    fig, ax = plt.subplots(figsize=(10,5))
    ax.grid()
    ax.set(xlabel = "Nombre d'epochs", ylabel="Loss", title="Evolution de la loss sur 200 epochs")
    plt.step(np.arange(len(loss_tab)), loss_tab)
    plt.savefig("loss_toy")

