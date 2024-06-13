import numpy as np
import data
import matplotlib.pyplot as plt


def fcann2_train(X, Y_):
    param_niter= int(1e5)
    param_delta=0.05
    param_lambda=1e-3
    hidden_size = 5
    C = max(Y_) + 1
    N = len(X)
    #Y_ = Y_.astype(int)
    # X     N x D
    # Y_    N x C
    Y_mat = np.zeros((N, C))
    for i in range(N):
        Y_mat[i][Y_[i]] = 1

    W1 =  np.random.randn(2, hidden_size)      
    W2 = np.random.randn(hidden_size, C)
    b1 = np.random.rand(hidden_size)
    b2 = np.random.rand(C)

    for i in range(param_niter):
        s1 = X @ W1 + b1            # N X hidden_size
        h1 = np.maximum(0, s1)      
        s2 = h1 @ W2 + b2           # hidden_size x C

        expscores = np.exp(s2 - np.max(s2)) # N x C
        sumexp = np.sum(expscores, axis=1).reshape((X.shape[0], 1))    # N x 1
        probs = expscores/sumexp  # N x C
        logprobs = np.log(probs)  # N x C

        # gubitak
        loss  = -1/N*sum(np.take_along_axis(logprobs, Y_[:,None], axis=1))     # scalar
        
        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # dL_dW2 = np.dot((probs - Y_mat).T, h1)
        # dL_db2 = (probs - Y_mat).T
        # dL_ds1 = (probs - Y_mat) @ W2.T @ np.diag(s1 > 0)
        # dL_dW1 = dL_ds1 @ X
        # dL_db1 = dL_ds1

        Gs2 = probs - Y_mat
        grad_W2 = 1/N * np.dot(h1.T, Gs2)
        grad_b2 = 1/N * np.sum(Gs2, axis=0)

        Gs1 = np.dot(np.dot((probs - Y_mat), W2.T), np.diag(np.diag(h1 > 0))) 
        grad_W1 = 1/N *np.dot(X.T, Gs1)
        grad_b1 = 1/N *np.sum(Gs1, axis=0)


        W2 += -param_delta * grad_W2
        b2 += -param_delta * grad_b2
        W1 += -param_delta * grad_W1
        b1 += -param_delta * grad_b1

    

    return W1, b1, W2, b2


def fcann2_classify(X, W1, b1, W2, b2):
    s1 = X @ W1 + b1            # N X hidden_size
    h1 = np.maximum(0, s1)      
    s2 = h1 @ W2 + b2           # hidden_size x C

    expscores = np.exp(s2 - np.max(s2)) # N x C
    sumexp = np.sum(expscores, axis=1).reshape((X.shape[0], 1))    # N x 1
    probs = expscores/sumexp     # N x C
    return probs


def fcann2_decfun(W1, b1, W2, b2):
    def classify(X):
      return np.argmax(fcann2_classify(X, W1, b1, W2, b2), axis=1)
    return classify


if __name__=="__main__":
    #np.random.seed(100)
    X,Y_ = data.sample_gmm_2d(6, 2, 10)
    X = (X - X.mean(0))/X.std(0)
    # train the model
    W1, b1, W2, b2 = fcann2_train(X, Y_)

    probs = fcann2_classify(X, W1, b1, W2, b2)

    Y = np.argmax(probs, axis=1)

    accuracy, pr, conf_mat = data.eval_perf_multi(Y,Y_)

    print (accuracy, pr)
    print(conf_mat)

    decfun = fcann2_decfun(W1, b1, W2, b2)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    data.graph_data(X, Y_, Y)

    # show the plot
    plt.show()