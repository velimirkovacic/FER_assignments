import torch
import torch.nn as nn
import torch.nn.functional as F


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        self.append(nn.BatchNorm2d(num_maps_in))
        self.append(nn.ReLU())
        self.append(nn.Conv2d(num_maps_in, num_maps_out, k, bias=bias))


class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        self.conv1 = _BNReluConv(input_channels, emb_size)
        self.conv2 = _BNReluConv(emb_size, emb_size)
        self.conv3 = _BNReluConv(emb_size, emb_size)
        self.maxpool = nn.MaxPool2d(3, 2)
        self.globalmaxpool = nn.AdaptiveAvgPool2d((1,1))

    def get_features(self, img):
        # Returns tensor with dimensions BATCH_SIZE, EMB_SIZE
        x = self.conv1(img)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.globalmaxpool(x)
        x = x.view(x.size(0), -1)
        return x

    def loss(self, anchor, positive, negative):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)

        loss = torch.sum(torch.relu(torch.norm(a_x - p_x, dim=1) - torch.norm(a_x - n_x, dim=1) + 1))
        return loss



class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def get_features(self, img):
        # YOUR CODE HERE
        feats = img.view(img.size(0), -1)
        return feats