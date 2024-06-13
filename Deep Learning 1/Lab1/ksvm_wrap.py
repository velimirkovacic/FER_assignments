import sklearn.svm as svm
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import data
import matplotlib.pyplot as plt


class KSVMWrap():
    '''
    Metode:
    __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        Konstruira omotač i uči RBF SVM klasifikator
        X, Y_:           podatci i točni indeksi razreda
        param_svm_c:     relativni značaj podatkovne cijene
        param_svm_gamma: širina RBF jezgre

    predict(self, X)
        Predviđa i vraća indekse razreda podataka X

    get_scores(self, X):
        Vraća klasifikacijske mjere
        (engl. classification scores) podataka X;
        ovo će vam trebati za računanje prosječne preciznosti.

    support
        Indeksi podataka koji su odabrani za potporne vektore
    '''
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.clf = svm.SVC(C=param_svm_c, gamma=param_svm_gamma, kernel="rbf")
        self.clf.fit(X, Y_)
        self.X = X
        self.Y_ = Y_
    
    def predict(self, X):
        return self.clf.predict(X)

    def support(self):
        indeksi = []
        for i in range(len(self.X)):
            if self.X[i] in self.clf.support_vectors_:
                indeksi += [i]
        return indeksi
    
    def get_scores(self):
        Y = self.predict(self.X)
        acc, pr, conf_mat = data.eval_perf_multi(Y, self.Y_)
        return acc, pr


if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    X,Y_ = data.sample_gmm_2d(6, 2, 10)
    
    model = KSVMWrap(X, Y_)
    Y = model.predict(X)
    special = model.support()
    acc, pr = model.get_scores()
    print(pr)
    print(acc)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(model.predict, bbox, offset=0.5)
    data.graph_data(X, Y_, Y, special)
    plt.show()