import sklearn.svm as svm
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
import data
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.optim.lr_scheduler as lr_scheduler
import modified_gpu.pt_deep_gpu as pt_deep



# Značajno brže pada pogreška
def train_mb(model, X, Yoh_, n, param_niter, param_delta, param_lambda):
    """Arguments:
        - X: model inputs [NxD], type: torch.Tensor
        - Yoh_: ground truth [NxC], type: torch.Tensor
        - param_niter: number of training iterations
        - param_delta: learning rate
    """
    optimizer = optim.SGD(model.parameters(), lr=param_delta)
    model.train()
    losses = []

    for i in range(param_niter):
        # Shufflanje
        indices = torch.randperm(X.shape[0])
        X=X[indices]
        Yoh_=Yoh_[indices]
        step = len(X)//n
        for j in range(0, len(X), step):
            X_mb = X[j - step:j]
            Yoh_mb = Yoh_[j - step:j]
            loss = model.get_loss(X_mb, Yoh_mb, param_lambda)

            losses += [loss]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i%100 == 0:
            print("iter:",i , "Loss:", loss)

    return losses


def train_mb_adam(model, X, Yoh_, n, param_niter, param_delta, param_lambda):
    """Arguments:
        - X: model inputs [NxD], type: torch.Tensor
        - Yoh_: ground truth [NxC], type: torch.Tensor
        - param_niter: number of training iterations
        - param_delta: learning rate
    """
    optimizer = optim.SGD(model.parameters(), lr=param_delta)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=1-1e-4)

    model.train()
    losses = []

    for i in range(param_niter):
        # Shufflanje
        indices = torch.randperm(X.shape[0])
        X=X[indices]
        Yoh_=Yoh_[indices]
        step = len(X)//n
        for j in range(0, len(X), step):
            X_mb = X[j - step:j]
            Yoh_mb = Yoh_[j - step:j]
            loss = model.get_loss(X_mb, Yoh_mb, param_lambda)

            losses += [loss]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if i%100 == 0:
            print("iter:",i , "Loss:", loss)
        scheduler.step()

    return losses


device = "cuda"
dataset_root = '/tmp/mnist'  # change this to your preference
mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

x_train, y_train = mnist_train.data, mnist_train.targets
x_test, y_test = mnist_test.data, mnist_test.targets
x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

N = x_train.shape[0]
D = x_train.shape[1] * x_train.shape[2]
C = y_train.max().add_(1).item()

y_train_oh = data.class_to_onehot(y_train)

device = "cpu"

x_train = x_train.flatten(1, -1).to(device)
x_test = x_test.flatten(1, -1).to(device)
y_train_oh = torch.from_numpy(y_train_oh).to(device)

#model = pt_deep.PTDeep([784, 100, 10])
#train_mb_adam(model, x_train, y_train_oh, 20, 4000, 0.05, 0)
#train_mb(model, x_train, y_train_oh, 20, 4000, 0.05, 0)


model = svm.SVC(C=1, gamma="auto", kernel="rbf")
model.fit(x_train, y_train)

y_train_pred = model.predict(x_train)
acc, pr, conf_mat = data.eval_perf_multi(y_train, y_train_pred)
print("Train:")
print(acc)
print(pr)
print(conf_mat)

y_test_pred = model.predict(x_test)
acc, pr, conf_mat = data.eval_perf_multi(y_test, y_test_pred)
print("Test:")
print(acc)
print(pr)
print(conf_mat)

# y_train_pred = np.argmax(model.forward(x_train).numpy(force=True), axis=1)
# acc, pr, conf_mat = data.eval_perf_multi(y_train.numpy(force=True), y_train_pred)
# print("Train:")
# print(acc)
# print(pr)
# print(conf_mat)

# y_test_pred = np.argmax(model.forward(x_test).numpy(force=True), axis=1)
# acc, pr, conf_mat = data.eval_perf_multi(y_test.numpy(force=True), y_test_pred)
# print("Test:")
# print(acc)
# print(pr)
# print(conf_mat)


# # Slike su vektorizirane
# x_train = x_train.flatten(1, -1).to(device)
# x_test = x_test.flatten(1, -1).to(device)
# y_train_oh = torch.from_numpy(y_train_oh).to(device)


# # for con, delta in [([784, 10], 0.1), ([784, 100, 10], 0.05), ([784, 100, 100, 10], 0.01), ([784, 100, 100, 100, 10], 0.005)]:
# #     print("Configuration:", con, "Delta:", delta)

#     model = pt_deep.PTDeep(con)
#     losses = pt_deep.train(model, x_train, y_train_oh, 10000, delta, 0)
#     tmp = []
#     for loss in losses:
#         tmp += [loss.numpy(force=True)]

#     losses = tmp

#     y_train_pred = np.argmax(model.forward(x_train).numpy(force=True), axis=1)
#     acc, pr, conf_mat = data.eval_perf_multi(y_train.numpy(force=True), y_train_pred)
#     print("Train:")
#     print(acc)
#     print(pr)
#     print(conf_mat)

#     y_test_pred = np.argmax(model.forward(x_test).numpy(force=True), axis=1)
#     acc, pr, conf_mat = data.eval_perf_multi(y_test.numpy(force=True), y_test_pred)
#     print("Test:")
#     print(acc)
#     print(pr)
#     print(conf_mat)


# from sklearn.model_selection import train_test_split


# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20)
# y_train_oh = data.class_to_onehot(y_train)
# y_val_oh = data.class_to_onehot(y_val)

# x_train = x_train.flatten(1, -1).to(device)
# x_test = x_test.flatten(1, -1).to(device)
# x_val = x_val.flatten(1, -1).to(device)
# y_train_oh = torch.from_numpy(y_train_oh).to(device)
# y_val_oh = torch.from_numpy(y_val_oh).to(device)
# model = pt_deep.PTDeep([784, 100, 10])
# min_model = pt_deep.train(model, x_train, y_train_oh, x_val, y_val_oh,  40000, 0.05, 0)

# y_train_pred = np.argmax(model.forward(x_train).numpy(force=True), axis=1)
# acc, pr, conf_mat = data.eval_perf_multi(y_train.numpy(force=True), y_train_pred)
# print("Train:")
# print(acc)
# print(pr)
# print(conf_mat)

# y_test_pred = np.argmax(model.forward(x_test).numpy(force=True), axis=1)
# acc, pr, conf_mat = data.eval_perf_multi(y_test.numpy(force=True), y_test_pred)
# print("Test:")
# print(acc)
# print(pr)
# print(conf_mat)


# y_train_pred = np.argmax(min_model.forward(x_train).numpy(force=True), axis=1)
# acc, pr, conf_mat = data.eval_perf_multi(y_train.numpy(force=True), y_train_pred)
# print("Train:")
# print(acc)
# print(pr)
# print(conf_mat)

# y_test_pred = np.argmax(min_model.forward(x_test).numpy(force=True), axis=1)
# acc, pr, conf_mat = data.eval_perf_multi(y_test.numpy(force=True), y_test_pred)
# print("Test:")
# print(acc)
# print(pr)
# print(conf_mat)


#plt.plot(range(len(losses)), losses)
#plt.show()
#plt.imshow(x_train[51], cmap = plt.get_cmap('gray'))
#plt.show()