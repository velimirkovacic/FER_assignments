import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import data
import matplotlib.pyplot as plt
import copy

class PTDeep(nn.Module):
    def __init__(self, config, activation="relu"):
        """Arguments:
 
        """
        super().__init__()
        self.W = []
        self.b = []
        self.activation = activation
        self.layers = len(config) - 1
        for i in range(len(config) - 1):
            self.W += [nn.Parameter(torch.from_numpy(np.random.randn(config[i] * config[i+1]).reshape(config[i], config[i+1]).astype("float32")), requires_grad=True)]
            self.b += [nn.Parameter(torch.tensor(np.random.randn(config[i+1]).astype("float32")), requires_grad=True)]
        self.W = nn.ParameterList(self.W).to("cuda")
        self.b = nn.ParameterList(self.b).to("cuda")


    def forward(self, X):
        for i in range(self.layers - 1):
            if self.activation == "softmax":
                scores = X.mm(self.W[i]) + self.b[i]
                X = torch.softmax(scores, dim=1)

            elif self.activation == "relu":
                scores = X.mm(self.W[i]) + self.b[i]
                X = torch.relu(scores)
        
        # Zadnji sloj je softmax
        scores = X.mm(self.W[self.layers - 1]) + self.b[self.layers - 1]
        X = torch.softmax(scores, dim=1)
        return X
    

    def get_loss(self, X, Yoh_, param_lambda):
            Y = self.forward(X)
            Y = torch.log(Y + 1e-10) * Yoh_
            Y = torch.sum(Y, dim=1)

            W = torch.linalg.norm(self.W[0])
            for i in range(1, self.layers): 
                W = W + torch.linalg.norm(self.W[i]) 

            return -torch.mean(Y) + param_lambda*W


def train(model, X, Yoh_, x_val, y_val_oh, param_niter, param_delta, param_lambda):
    """Arguments:
        - X: model inputs [NxD], type: torch.Tensor
        - Yoh_: ground truth [NxC], type: torch.Tensor
        - param_niter: number of training iterations
        - param_delta: learning rate
    """
    optimizer = optim.SGD(model.parameters(), lr=param_delta)
    model.train()
    losses = []
    min_loss_val = torch.inf
    min_loss_it = 0
    min_model = None
    for i in range(param_niter):
        loss = model.get_loss(X, Yoh_, param_lambda)
        loss_val = model.get_loss(x_val, y_val_oh, param_lambda)

        if loss_val < min_loss_val:
            min_loss_val = loss_val
            min_loss_it = i
            min_model = copy.deepcopy(model)


        # if i > 10 and torch.abs(loss - losses[-1]) < 1e-4:
        #     print("Stopping at iteration", i, "(loss change smaller than 1e-4)")
        #     break

        losses += [loss]
        if i%1000 == 0:
            print("iter:",i , "Loss:", loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return min_model


def softmax_decfun():
    def sm(x):
        return torch.softmax(x, dim=1)
    return sm


if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    X,Y_ = data.sample_gmm_2d(6, 2, 10)
    X = (X - X.mean(0))/X.std(0)
    Yoh_ = np.zeros((6*10, 2))
    for i in range(6*10):
        Yoh_[i][Y_[i]] = 1
    
    ptlr = PTDeep([2, 10, 10, 2], "softmax")
    train(ptlr, X, Yoh_, 10000, 0.1, 0)

