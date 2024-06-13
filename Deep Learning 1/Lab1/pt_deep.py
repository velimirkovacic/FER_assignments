import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import data
import matplotlib.pyplot as plt

class PTDeep(nn.Module):
    def __init__(self, config, activation):
        """Arguments:
 
        """
        super().__init__()
        self.activation = activation
        self.W = []
        self.b = []
        self.layers = len(config) - 1
        for i in range(len(config) - 1):
            self.W += [nn.Parameter(torch.from_numpy(np.random.randn(config[i] * config[i+1]).reshape(config[i], config[i+1])), requires_grad=True)]
            self.b += [nn.Parameter(torch.tensor(np.random.randn(config[i+1])), requires_grad=True)]
        self.W = nn.ParameterList(self.W)
        self.b = nn.ParameterList(self.b)


    def forward(self, X_np):
        X = torch.tensor(X_np)
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
    

    def get_loss(self, X_np, Yoh_np, param_lambda):
        X = torch.tensor(X_np)
        Yoh_ = torch.tensor(Yoh_np)
        Y = self.forward(X)
        Y = torch.log(Y + 1e-10) * Yoh_
        Y = torch.sum(Y, dim=1)

        W = torch.linalg.norm(self.W[0])
        for i in range(1, self.layers): 
            W = W + torch.linalg.norm(self.W[i]) 

        return -torch.mean(Y) + param_lambda*W


def train(model, X_np, Yoh_np, param_niter, param_delta, param_lambda):
    """Arguments:
        - X: model inputs [NxD], type: torch.Tensor
        - Yoh_: ground truth [NxC], type: torch.Tensor
        - param_niter: number of training iterations
        - param_delta: learning rate
    """
    X = torch.tensor(X_np)
    Yoh_ = torch.tensor(Yoh_np)
    optimizer = optim.SGD(model.parameters(), lr=param_delta)
    model.train()
    losses = []
    for i in range(param_niter):
        loss = model.get_loss(X, Yoh_, param_lambda)
        losses += [loss]
        print("iter:",i , "Loss:", loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses

def eval(model, X_np):
    """Arguments:
        - model: type: PTLogreg
        - X: actual datapoints [NxD], type: np.array
        Returns: predicted class probabilites [NxC], type: np.array
    """
    X = torch.from_numpy(X_np)
    Y = model.forward(X).detach().numpy()
    return np.argmax(Y, axis=1)


def count_params(model):
    cnt = 0
    for param in model.named_parameters():
        print(param[0].strip("Parameter containing:"), param[1].nelement())
        cnt += param[1].nelement()
    return cnt


def eval_decfun(ptdeep):
    def classify(X):
      return eval(ptdeep, X)
    return classify

def softmax_decfun():
    def sm(x):
        return torch.softmax(x, dim=1)
    return sm

def relu_decfun():
    def relu(x):
        return nn.functional.relu(x)
    return relu

if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    X,Y_ = data.sample_gmm_2d(6, 2, 10)
    X = (X - X.mean(0))/X.std(0)
    Yoh_ = np.zeros((6*10, 2))
    for i in range(6*10):
        Yoh_[i][Y_[i]] = 1
    torch.autograd.set_detect_anomaly(True)
    # definiraj model:
    ptdeep = PTDeep([2, 10, 10, 2], "relu")
    count_params(ptdeep)
    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(ptdeep, X, Yoh_, 10000, 0.1, 0)

    # dohvati vjerojatnosti na skupu za učenje
    Y_pred = eval(ptdeep, X)

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, pr, conf_mat = data.eval_perf_multi(Y_pred,Y_)

    print (accuracy, pr)
    print(conf_mat)

    # iscrtaj rezultate, decizijsku plohu

    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(eval_decfun(ptdeep), bbox, offset=0.5)
    data.graph_data(X, Y_, Y_pred)


    plt.show()