import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context


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

writer = SummaryWriter()
for n_iter in range(100):
    ##  TODO:  Calcul du forward (loss)
    yhat = linear.forward(ctx, x, w, b)
    loss = mse.forward(ctx2, yhat, y)
    #print("L : ", loss)
    # `loss` doit correspondre au coût MSE calculé à cette itération
    # on peut visualiser avec
    # tensorboard --logdir runs/
    #writer.add_scalar('Loss/train', loss, n_iter)

    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")

    ##  TODO:  Calcul du backward (grad_w, grad_b)
    _, grad_w, grad_b = linear.backward(ctx, yhat)
    # print(grad_w.shape, grad_b.shape)

    ##  TODO:  Mise à jour des paramètres du modèle
    w = w - epsilon*grad_w
    b = b - epsilon*grad_b

