import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import data
import matplotlib.pyplot as plt

class PTLogreg(nn.Module):
    def __init__(self, D, C):
        """Arguments:
        - D: dimensions of each datapoint 
        - C: number of classes
        """
        super().__init__()
        self.W = torch.nn.Parameter(torch.from_numpy(np.random.randn(C * D).reshape(C, D)), requires_grad=True)
        self.b = torch.nn.Parameter(torch.zeros(C), requires_grad=True)

    def forward(self, X_np):
        X = torch.tensor(X_np)
        scores = torch.mm(X, self.W.T) + self.b
        Y = torch.softmax(scores, 1)
        return Y
    

    def get_loss(self, X_np, Yoh_np, param_lambda):
        X = torch.tensor(X_np)
        Yoh_ = torch.tensor(Yoh_np)
        Y = self.forward(X)
        Y = torch.log(Y) * Yoh_
        Y = torch.sum(Y, dim=1)
        
        return -torch.mean(Y) + param_lambda*torch.linalg.norm(self.W)


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
    for i in range(param_niter):
        loss = model.get_loss(X, Yoh_, param_lambda)
        print("Loss:", loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def eval(model, X_np):
    """Arguments:
        - model: type: PTLogreg
        - X: actual datapoints [NxD], type: np.array
        Returns: predicted class probabilites [NxC], type: np.array
    """
    X = torch.from_numpy(X_np)
    Y = model.forward(X).detach().numpy()
    return np.argmax(Y, axis=1)




def eval_decfun(ptlr):
    def classify(X):
      return eval(ptlr, X)
    return classify






if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    X,Y_ = data.sample_gauss_2d(3, 100)
    Yoh_ = np.zeros((100*3, 3))
    for i in range(100*3):
        Yoh_[i][Y_[i]] = 1
    
    # definiraj model:
    ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])
    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(ptlr, X, Yoh_, 1000, 0.5, 0)

    # dohvati vjerojatnosti na skupu za učenje
    Y_pred = eval(ptlr, X)

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, pr, conf_mat = data.eval_perf_multi(Y_pred,Y_)

    print (accuracy, pr)
    print(conf_mat)

    # iscrtaj rezultate, decizijsku plohu

    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(eval_decfun(ptlr), bbox, offset=0.5)
    data.graph_data(X, Y_, Y_pred)


    plt.show()