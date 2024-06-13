import numpy as np
import data
import matplotlib.pyplot as plt
def logreg_train(X, Y_):
    C = max(Y_) + 1
    '''
    Argumenti
        X:  podatci, np.array NxD
        Y_: indeksi razreda, np.array Nx1

    Povratne vrijednosti
        W, b: parametri logističke regresije
    '''
    param_niter = 1000
    param_delta = 0.1
    W = np.random.randn(C * X.shape[1]).reshape(C, X.shape[1])
    b = 0
    N = X.shape[0]
    C_mat = np.array([list(range(0, C))] * N)
    Y_mat = np.zeros((N, C))
    for i in range(N):
        Y_mat[i][Y_[i]] = 1
    #Y_mat = np.zeros((N, C)) - np.take_along_axis(np.ones((N, C)), Y_[:,None], axis=1)
    for i in range(param_niter):
        # eksponencirane klasifikacijske mjere
        # pri računanju softmaksa obratite pažnju
        # na odjeljak 4.1 udžbenika
        # (Deep Learning, Goodfellow et al)!
        scores = np.dot(X, W.T) + b    # N x C
        expscores = np.exp(scores - np.max(scores)) # N x C
        # nazivnik sofmaksa
        sumexp = np.sum(expscores, axis=1).reshape((X.shape[0], 1))    # N x 1

        # logaritmirane vjerojatnosti razreda 
        probs = expscores/sumexp     # N x C
        logprobs = np.log(probs)  # N x C

        # gubitak
        loss  = -1/N*sum(np.take_along_axis(logprobs, Y_[:,None], axis=1))     # scalar
        
        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije komponenata gubitka po mjerama
        dL_ds = probs - Y_mat     # N x C

        # gradijenti parametara
        grad_W = 1/N * np.dot(dL_ds.T, X)    # C x D (ili D x C)
        grad_b = 1/N * np.sum(dL_ds, axis=0)    # C x 1 (ili 1 x C)
        # poboljšani parametri
        W += -param_delta * grad_W
        b += -param_delta * grad_b
    return W, b

def binlogreg_classify(X, W, b):
    '''
    Argumenti
        X:    podatci, np.array NxD
        w, b: parametri logističke regresije 

    Povratne vrijednosti
        probs: vjerojatnosti razreda c1
    '''
    scores = np.dot(X, W.T) + b    # N x C
    expscores = np.exp(scores - np.max(scores)) # N x C
    # nazivnik sofmaksa
    sumexp = np.sum(expscores, axis=1).reshape((X.shape[0], 1))    # N x 1

    # logaritmirane vjerojatnosti razreda 
    probs = expscores/sumexp     # N x C
    return probs

def logreg_decfun(W,b):
    def classify(X):
      return np.argmax(binlogreg_classify(X, W,b), axis=1)
    return classify



if __name__=="__main__":
    #np.random.seed(200)

    # get the training dataset
    X,Y_ = data.sample_gauss_2d(3, 100)

    # train the model
    W,b = logreg_train(X, Y_)

    probs = binlogreg_classify(X, W, b)

    Y = np.argmax(probs, axis=1)

    accuracy, conf_mat, precision, recall = data.eval_perf_multi(Y,Y_)

    print (accuracy, recall, precision)
    print(conf_mat)

    decfun = logreg_decfun(W,b)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    data.graph_data(X, Y_, Y)

    # show the plot
    plt.show()