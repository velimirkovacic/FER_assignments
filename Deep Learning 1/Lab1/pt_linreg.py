import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

## Definicija računskog grafa
# podaci i parametri, inicijalizacija parametara
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

X = torch.tensor([[1, 3], [2, 5], [3, 7], [4, 8]])

# optimizacijski postupak: gradijentni spust
optimizer = optim.SGD([a, b], lr=0.1)

N = len(X)
Y = X[:,1]
X = X[:,0]
for i in range(100):
    # afin regresijski model
    Y_ = a*X + b
    print(Y_)
    diff = Y - Y_

    # kvadratni gubitak
    loss = 1/N * torch.sum(torch.abs(diff))

    # računanje gradijenata
    loss.backward()

    grad_a = -2 * torch.mean(diff * X)
    grad_b = -2 * torch.mean(diff)
    print("grad a", grad_a, a.grad)
    print("grad b", grad_b, b.grad)

    # korak optimizacije
    optimizer.step()

    # Postavljanje gradijenata na nulu
    optimizer.zero_grad()

    print(f'step: {i}, loss:{loss}, a:{a}, b {b}')



a = a.detach().numpy()[0]
b = b.detach().numpy()[0]
x = np.arange(0, 5, 0.1)
plt.scatter(X, Y)

plt.plot(x, a*x + b)
plt.show()
    