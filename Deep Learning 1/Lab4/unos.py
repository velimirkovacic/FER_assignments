from torch.utils.data import Dataset
from collections import defaultdict
from random import choice
import torchvision
import random
import torch

class MNISTMetricDataset(Dataset):
    def __init__(self, root="/tmp/mnist/", split='train', remove_class=None):
        super().__init__()
        assert split in ['train', 'test', 'traineval']
        self.root = root
        self.split = split
        mnist_ds = torchvision.datasets.MNIST(self.root, train='train' in split, download=True)
        self.images, self.targets = mnist_ds.data.float() / 255., mnist_ds.targets
        self.classes = list(range(10))

        self.size = len(self.images)


        if remove_class is not None:
            images_new, targets_new = [], []

            for i in range(len(self.images)):
                if self.targets[i] != remove_class:
                    images_new.append(self.images[i])
                    targets_new.append(self.targets[i])

            self.images = images_new
            self.targets = targets_new

            self.size = len(self.images)


        self.target2indices = defaultdict(list)
        for i in range(len(self.images)):
            self.target2indices[self.targets[i].item()] += [i]


    def _sample_negative(self, index):
        index_new = random.randint(1, self.size - 1)

        while self.targets[index_new] == self.targets[index]:
            index_new = random.randint(1, self.size - 1)
        
        return index_new

    def _sample_positive(self, index):
        index_new = random.randint(1, self.size - 1)
        
        while self.targets[index_new] != self.targets[index]:
            index_new = random.randint(1, self.size - 1)

        return index_new


    def __getitem__(self, index):
        anchor = self.images[index].unsqueeze(0)
        target_id = self.targets[index].item()
        if self.split in ['traineval', 'val', 'test']:
            return anchor, target_id
        else:
            positive = self._sample_positive(index)
            negative = self._sample_negative(index)
            positive = self.images[positive]
            negative = self.images[negative]
            return anchor, positive.unsqueeze(0), negative.unsqueeze(0), target_id

    def __len__(self):
        return len(self.images)
    

if __name__ == "__main__":
    ds = MNISTMetricDataset()