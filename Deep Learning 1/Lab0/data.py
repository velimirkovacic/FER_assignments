import numpy as np
import matplotlib.pyplot as plt

class Random2DGaussian:

    def __init__(self, minx = 0, maxx = 10, miny = 0, maxy = 10):
        mi = (maxx - minx) * np.random.random_sample(2) + minx

        self.mi = np.array(mi)

        eigvalx = (np.random.random_sample()*(maxx - minx)/5)**2
        eigvaly = (np.random.random_sample()*(maxy - miny)/5)**2

        D = np.array([[eigvalx, 0],[0, eigvaly]])
        
        fi = 360 * np.random.random_sample()
        
        R = np.array([[np.cos(fi), -np.sin(fi)],[np.sin(fi), np.cos(fi)]])
        
        self.cov = R.T * D * R


    def get_sample(self, n):
        return np.random.multivariate_normal(self.mi, self.cov, n)

def eval_perf_binary(Y,Y_):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(Y_)):
        if Y_[i] == 1:
            if Y[i] == 1:
                TP += 1
            else:
                FN += 1
        else:
            if Y[i] == 1:
                FP += 1
            else:
                TN += 1    
    Acc = (TP + TN)/len(Y_)
    P = TP/(TP + FP)
    R = TP/(TP + FN)

    return Acc, P, R

def Preciznost(Yr, i):
    Ysplit = np.array([0] * i + [1] * (len(Yr) - i))
    _, P, _ = eval_perf_binary(Ysplit, Yr)
    return P

def eval_AP(Yr):
    brojnik = 0
    nazivnik = 0
    for i in range(len(Yr)):
        brojnik += Preciznost(Yr, i) * Yr[i]
        nazivnik += Yr[i]
    return brojnik/nazivnik

def sample_gmm_2d(K, C, N):
    X = None
    Y = None
    for i in range(K):
        c = np.random.randint(0, C)
        G = Random2DGaussian()
        x = G.get_sample(N)
        y = np.ones(N) * c
        if i == 0:
            X = x
            Y = y
        else:
            X = np.concatenate((X, x), 0)
            Y = np.concatenate((Y, y), 0)
    return X,Y.astype(int)
        
def sample_gauss_2d(C, N):
    X = None
    Y = None
    for i in range(C):
        G = Random2DGaussian()
        x = G.get_sample(N)
        y = np.ones(N) * i
        if i == 0:
            X = x
            Y = y
        else:
            X = np.concatenate((X, x), 0)
            Y = np.concatenate((Y, y), 0)
    return X,Y.astype(int)
    
def sample_gauss_2d(C, N):
    X = None
    Y = None
    for i in range(C):
        G = Random2DGaussian()
        x = G.get_sample(N)
        y = np.ones(N) * i
        if i == 0:
            X = x
            Y = y
        else:
            X = np.concatenate((X, x), 0)
            Y = np.concatenate((Y, y), 0)
    return X,Y.astype(int)
        
def graph_data(X, Y_, Y):
    '''
    X  ... podatci (np.array dimenzija Nx2)
    Y_ ... točni indeksi razreda podataka (Nx1)
    Y  ... predviđeni indeksi razreda podataka (Nx1)
    '''
    palette=([0.5,0.5,0.5], [1,1,1], [0.2,0.2,0.2])
    colors = np.tile([0.0,0.0,0.0], (Y_.shape[0],1))
    for i in range(len(palette)):
        colors[Y_==i] = palette[i]

    good = (Y_ == Y)
    plt.scatter(X[good, 0], X[good, 1], marker = "o", c=colors[good], edgecolors="black")

    bad = (Y_ != Y)
    plt.scatter(X[bad, 0], X[bad, 1], marker="s", c=colors[bad], edgecolors="black")


def graph_surface(fun, rect, offset=0.5, width=256, height=256):
    '''
    fun    ... decizijska funkcija (Nx2)->(Nx1)
    rect   ... željena domena prikaza zadana kao:
                ([x_min,y_min], [x_max,y_max])
    offset ... "nulta" vrijednost decizijske funkcije na koju 
                je potrebno poravnati središte palete boja;
                tipično imamo:
                offset = 0.5 za probabilističke modele 
                    (npr. logistička regresija)
                offset = 0 za modele koji ne spljošćuju
                    klasifikacijske mjere (npr. SVM)
    width,height ... rezolucija koordinatne mreže
    '''
    X = np.arange(rect[0][0], rect[1][0], (rect[1][0] - rect[0][0])/width)
    Y = np.arange(rect[0][1], rect[1][1], (rect[1][1] - rect[0][1])/width)

    
    xx, yy = np.meshgrid(X,Y)
    print(X.shape, Y.shape, xx.shape, yy.shape)
    mesh = np.stack((xx.flatten(), yy.flatten()), axis=1)
    print(xx.flatten().shape, yy.flatten().shape)
    print(mesh.shape)
    res = fun(mesh).reshape((width, height))

    delta = offset 

    maxval = max(np.max(res) - delta, - (np.min(res) - delta))


    plt.pcolormesh(xx, yy, res, vmin=delta-maxval,vmax=delta+maxval)
    plt.contour(xx, yy, res, colors="black", levels=[offset])


def eval_perf_multi(Y,Y_):
    #acc, conf mat, prec, recall
    C = max(Y_) + 1
    conf_mat = np.zeros((C,C))

    for i in range(len(Y_)):
        conf_mat[Y[i]][Y_[i]] += 1

    acc = np.sum(np.diag(conf_mat))/np.sum(conf_mat)

    prec = []
    rec = []

    for i in range(C):
        TP = conf_mat[i][i]
        FP = np.sum(conf_mat[i]) - conf_mat[i][i]
        FN = np.sum(conf_mat, axis=0)[i] - conf_mat[i][i]
        prec += [TP/(TP+FP)]
        rec += [TP/(TP+FN)]

    prec = sum(prec)/C
    rec = sum(rec)/C

    return acc, conf_mat, prec, rec

if __name__=="__main__":
    pass