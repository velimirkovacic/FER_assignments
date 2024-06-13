import torch
from torch import nn
import time
from pathlib import Path

import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from nn import draw_conv_filters
DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'out'
device = "cuda"



class CovolutionalModel(nn.Module):
  def __init__(self, in_channels, conv1_width, conv2_width, fc1_width, class_count):
    super(CovolutionalModel, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels=conv1_width, kernel_size=5, stride=1, padding=1, bias=True).to(device)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1).to(device)

    self.conv2 = nn.Conv2d(in_channels=conv1_width, out_channels=conv2_width, kernel_size=5, stride=1, padding=1, bias=True).to(device)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1).to(device)

    self.fc1 = nn.Linear(7*7*conv2_width, fc1_width, bias=True).to(device)

    self.fc_logits = nn.Linear(fc1_width, class_count, bias=True).to(device)

    self.reset_parameters()

  def reset_parameters(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear) and m is not self.fc_logits:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    self.fc_logits.reset_parameters()

  def forward(self, x):
    h = self.conv1(x)
    h = self.pool1(h)
    h = torch.relu(h) 
    h = self.conv2(h)
    h = self.pool2(h)
    h = torch.relu(h)
    h = h.view(h.shape[0], -1)
    h = self.fc1(h)
    h = torch.relu(h)
    logits = self.fc_logits(h)
    return logits


def train(train_x, train_y, valid_x, valid_y, model, config):
    lr_policy = config['lr_policy']
    weight_decay = config['weight_decay']
    batch_size = config['batch_size']
    max_epochs = config['max_epochs']
    save_dir = config['save_dir']
    num_examples = train_x.shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size

    train_dataset = TensorDataset(train_x, train_y)
    valid_dataset = TensorDataset(valid_x, valid_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()


    for epoch in range(1, max_epochs+1):
        if epoch in lr_policy:
            solver_config = lr_policy[epoch]["lr"]
        cnt_correct = 0

        optimizer = optim.SGD(model.parameters(), lr=solver_config, weight_decay=weight_decay)

        model.train()

        for i, (batch_x, batch_y) in enumerate(train_loader):
            logits = model.forward(batch_x)
            loss = criterion(logits, batch_y)

            yp = torch.argmax(logits, 1)
            yt = torch.argmax(batch_y, 1)
            cnt_correct += (yp == yt).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 5 == 0:
                print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i*batch_size, num_examples, loss))
            if i % 100 == 0:
                draw_conv_filters(epoch, i*batch_size, next(model.conv1.parameters()).data, save_dir, tensor=True)
            if i > 0 and i % 50 == 0:
                print("Train accuracy = %.2f" % (cnt_correct / ((i+1)*batch_size) * 100))
        print("Train accuracy = %.2f" % (cnt_correct / num_examples * 100))
        evaluate("Validation", valid_dataset, model, config)
    return model

def evaluate(name, dataset, model, config):
    print("\nRunning evaluation: ", name)
    batch_size = config['batch_size']
    num_examples = len(dataset)
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size
    cnt_correct = 0
    loss_avg = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()


    for i, (batch_x, batch_y) in enumerate(loader):

        logits = model.forward(batch_x)
        loss = criterion(logits, batch_y)

        yp = torch.argmax(logits, 1)
        yt = torch.argmax(batch_y, 1)
        cnt_correct += (yp == yt).sum()
        loss = criterion(logits, batch_y)
        loss_avg += loss

    valid_acc = cnt_correct / num_examples * 100
    loss_avg /= num_batches
    print(name + " accuracy = %.2f" % valid_acc)
    print(name + " avg loss = %.2f\n" % loss_avg)


config = {}
config['max_epochs'] = 8
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['weight_decay'] = 1e-3
config['lr_policy'] = {1:{'lr':1e-1}, 3:{'lr':1e-2}, 5:{'lr':1e-3}, 7:{'lr':1e-4}}

def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]

np.random.seed(int(time.time() * 1e6) % 2**31)

ds_train, ds_test = MNIST(DATA_DIR, train=True, download=True), MNIST(DATA_DIR, train=False)
train_x = ds_train.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float32) / 255
train_y = ds_train.targets.numpy()
train_x, valid_x = train_x[:55000], train_x[55000:]
train_y, valid_y = train_y[:55000], train_y[55000:]
test_x = ds_test.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float32) / 255
test_y = ds_test.targets.numpy()
train_mean = train_x.mean()
train_x, valid_x, test_x = (x - train_mean for x in (train_x, valid_x, test_x))
train_y, valid_y, test_y = (dense_to_one_hot(y, 10) for y in (train_y, valid_y, test_y))

train_x = torch.tensor(train_x).to(device)
valid_x = torch.tensor(valid_x).to(device)
test_x = torch.tensor(test_x).to(device)
train_y = torch.tensor(train_y).to(device)
valid_y = torch.tensor(valid_y).to(device)
test_y = torch.tensor(test_y).to(device)

model = CovolutionalModel(1, 16, 32, 512, 10)


train(train_x, train_y, valid_x, valid_y, model, config)


test_dataset = TensorDataset(test_x, test_y)

evaluate("Test", test_dataset, model, config)

