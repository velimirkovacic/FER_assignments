import os
import pickle
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
import torch.optim as optim
import torch
import skimage as ski
import math
import matplotlib.pyplot as plt


class CovolutionalModel(nn.Module):
  # conv(16,5) -> relu() -> pool(3,2) -> conv(32,5) -> relu() -> pool(3,2) -> fc(256) -> relu() -> fc(128) -> relu() -> fc(10)
  def __init__(self):
    super(CovolutionalModel, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=1, bias=True).to(device)
    self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(device)

    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=1, bias=True).to(device)
    self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(device)

    self.fc1 = nn.Linear(7*7*32, 128, bias=True).to(device)
    self.fc_logits = nn.Linear(128, 10, bias=True).to(device)

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
    h = torch.relu(h)
    h = self.pool1(h)
    h = self.conv2(h)
    h = torch.relu(h) 
    h = self.pool2(h)
    h = h.view(h.shape[0], -1)
    h = self.fc1(h)
    h = torch.relu(h)
    logits = self.fc_logits(h)
    return logits

def draw_conv_filters(epoch, step, weights, save_dir):
  w = weights.copy()
  num_filters = w.shape[0]
  num_channels = w.shape[1]
  k = w.shape[2]
  assert w.shape[3] == w.shape[2]
  w = w.transpose(2, 3, 1, 0)
  w -= w.min()
  w /= w.max()
  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border
  img = np.zeros([height, width, num_channels])
  for i in range(num_filters):
    r = int(i / cols) * (k + border)
    c = int(i % cols) * (k + border)
    img[r:r+k,c:c+k,:] = w[:,:,:,i]
  filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
  img =(img * 255).astype(np.uint8)
  ski.io.imsave(os.path.join(save_dir, filename), img)

def plot_training_progress(save_dir, data):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

    linewidth = 2
    legend_size = 10
    train_color = 'm'
    val_color = 'c'

    num_points = len(data['train_loss'])
    x_data = np.linspace(1, num_points, num_points)
    ax1.set_title('Cross-entropy loss')
    ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
            linewidth=linewidth, linestyle='-', label='train')
    ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
            linewidth=linewidth, linestyle='-', label='validation')
    ax1.legend(loc='upper right', fontsize=legend_size)
    ax2.set_title('Average class accuracy')
    ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
            linewidth=linewidth, linestyle='-', label='train')
    ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
            linewidth=linewidth, linestyle='-', label='validation')
    ax2.legend(loc='upper left', fontsize=legend_size)
    ax3.set_title('Learning rate')
    ax3.plot(x_data, data['lr'], marker='o', color=train_color,
            linewidth=linewidth, linestyle='-', label='learning_rate')
    ax3.legend(loc='upper left', fontsize=legend_size)

    save_path = os.path.join(save_dir, 'training_plot.png')
    print('Plotting in: ', save_path)
    plt.savefig(save_path)

def plot_training_progress(save_dir, data):
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

  linewidth = 2
  legend_size = 10
  train_color = 'm'
  val_color = 'c'

  num_points = len(data['train_loss'])
  x_data = np.linspace(1, num_points, num_points)
  ax1.set_title('Cross-entropy loss')
  ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax1.legend(loc='upper right', fontsize=legend_size)
  ax2.set_title('Average class accuracy')
  ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax2.legend(loc='upper left', fontsize=legend_size)
  ax3.set_title('Learning rate')
  ax3.plot(x_data, data['lr'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='learning_rate')
  ax3.legend(loc='upper left', fontsize=legend_size)

  save_path = os.path.join(save_dir, 'training_plot.png')
  print('Plotting in: ', save_path)
  plt.savefig(save_path)


def evaluate(yp, yt, c):
    conf_mat = np.zeros((c, c)).astype(int)
    for i in range(len(yp)):
        conf_mat[yt[i]][yp[i]] += 1

    acc = conf_mat.trace() / conf_mat.sum()
    precision = conf_mat.diagonal() / conf_mat.sum(axis=1)
    recall = conf_mat.diagonal() / conf_mat.sum(axis=0)

    precision = np.nan_to_num(precision, 0)
    recall = np.nan_to_num(recall, 0)

    return conf_mat, acc, precision, recall
      

def train(train_x, train_y, valid_x, valid_y, model, config, plot_data):
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    batch_size = config['batch_size']
    max_epochs = config['max_epochs']
    num_examples = train_x.shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size

    train_dataset = TensorDataset(train_x, train_y)
    valid_dataset = TensorDataset(valid_x, valid_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader_valid_plot = DataLoader(valid_dataset, batch_size=len(valid_dataset))
    dataloader_train_plot = DataLoader(train_dataset, batch_size=len(train_dataset))


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(1, max_epochs+1):
        model.train()

        for i, (batch_x, batch_y) in enumerate(train_loader):

            logits = model.forward(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 5 == 0:
                print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i*batch_size, num_examples, loss))
            if i % 100 == 0:
                draw_conv_filters(epoch, i*batch_size, next(model.conv1.parameters()).data.cpu().detach().numpy(), SAVE_DIR)
            if i > 0 and i % 50 == 0:
                yp = torch.argmax(logits, 1)
                yt = torch.argmax(batch_y, 1)
                _, acc, _, _ = evaluate(yp,yt, 10)
                print("Train accuracy = %.2f" % acc)
      
        with torch.no_grad():
          batch_data, batch_targets = next(iter(dataloader_valid_plot))
          logits = model.forward(batch_data)
          yp = torch.argmax(logits, 1)
          yt = torch.argmax(batch_targets, 1)
          loss = criterion(logits, batch_targets)
          conf, acc, prec, rec = evaluate(yp,yt, 10)
          plot_data['valid_acc'] += [acc]
          plot_data['valid_loss'] += [loss.cpu().detach().numpy()]
          
          print(conf)
          print("Accuracy:\n", acc)
          print("Precision:\n", prec)
          print("Recall:\n", rec)

          batch_data, batch_targets = next(iter(dataloader_train_plot))
          logits = model.forward(batch_data)
          yp = torch.argmax(logits, 1)
          yt = torch.argmax(batch_targets, 1)
          loss = criterion(logits, batch_targets)
          conf, acc, prec, rec = evaluate(yp,yt, 10)
          plot_data['train_acc'] += [acc]
          plot_data['train_loss'] += [loss.cpu().detach().numpy()]
          
          plot_data['lr'] += [scheduler.get_lr()]


        scheduler.step()
    return plot_data

def shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y

def unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='latin1')
  fo.close()
  return dict

SAVE_DIR = Path(__file__).parent / 'out'
DATA_DIR = Path(__file__).parent / 'cifar'
device = "cuda"

img_height = 32
img_width = 32
num_channels = 3
num_classes = 10

train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
train_y = []
for i in range(1, 6):
  subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
  train_x = np.vstack((train_x, subset['data']))
  train_y += subset['labels']
train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
train_y = np.array(train_y, dtype=np.int32)

subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
test_y = np.array(subset['labels'], dtype=np.int32)

valid_size = 5000
train_x, train_y = shuffle_data(train_x, train_y)
valid_x = train_x[:valid_size, ...]
valid_y = train_y[:valid_size, ...]
train_x = train_x[valid_size:, ...]
train_y = train_y[valid_size:, ...]
data_mean = train_x.mean((0, 1, 2))
data_std = train_x.std((0, 1, 2))

train_x = (train_x - data_mean) / data_std
valid_x = (valid_x - data_mean) / data_std
test_x = (test_x - data_mean) / data_std

train_x = train_x.transpose(0, 3, 1, 2)
valid_x = valid_x.transpose(0, 3, 1, 2)
test_x = test_x.transpose(0, 3, 1, 2)

def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]
train_y, valid_y, test_y = (dense_to_one_hot(y, 10) for y in (train_y, valid_y, test_y))


train_x = torch.tensor(train_x).to(device)
train_y = torch.tensor(train_y).to(device)
test_x = torch.tensor(test_x).to(device)
test_y = torch.tensor(test_y).to(device)
valid_x = torch.tensor(valid_x).to(device)
valid_y = torch.tensor(valid_y).to(device)

model = CovolutionalModel()
config = {}
config['max_epochs'] = 50
config['batch_size'] = 50
config['weight_decay'] = 1e-3
config['learning_rate'] = 1e-1

plot_data = {}
plot_data['train_loss'] = []
plot_data['valid_loss'] = []
plot_data['train_acc'] = []
plot_data['valid_acc'] = []
plot_data['lr'] = []

plot_data = train(train_x, train_y, valid_x, valid_y, model, config, plot_data)

plot_training_progress(SAVE_DIR, plot_data)



test_dataset = TensorDataset(test_x, test_y)
dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))
batch_data, batch_targets = next(iter(dataloader))
logits = model.forward(batch_data)
yp = torch.argmax(logits, 1)
yt = torch.argmax(batch_targets, 1)
conf, acc, prec, rec = evaluate(yp,yt, 10)
print("Test set")
print(conf)
print("Accuracy:\n", acc)
print("Precision:\n", prec)
print("Recall:\n", rec)



# criterion = nn.CrossEntropyLoss()
# loss_vector = []
# for i in range(len(batch_data)):
#     loss = criterion(logits[i:i+1], batch_targets[i:i+1])
#     loss_vector.append(loss.item())

# loss_vector = torch.tensor(loss_vector)
# indexes = np.argsort(-loss_vector)[:10]
# batch_data = batch_data.cpu().detach().numpy()
# batch_targets = batch_targets.cpu().detach().numpy()
# loss_vector = loss_vector.cpu().detach().numpy()

# plt.clf()
# fig, ax = plt.subplots(2, 5, figsize=(15, 6))
# ax = ax.flatten()
# j = 0
# for i in indexes:
#   ax[j].imshow(np.transpose(batch_data[i], (1, 2, 0)))
#   ax[j].axis('off')
#   ax[j].set_title("Class: " + str(np.argmax(batch_targets[i])) + " Loss: " + format(loss_vector[i], f".{5}f"))
#   j += 1
# plt.show()