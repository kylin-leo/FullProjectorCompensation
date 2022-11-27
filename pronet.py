import numpy as np
import torch
from torch.nn.utils import spectral_norm
import torch.nn as nn
import torch.nn.functional as F
import scipy


class ProNet(torch.nn.Module):
    def __init__(self, img_rows=1024, img_cols=1536, num_primary=3, num_filter = 3*4):
        super(ProNet, self).__init__()

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_ranks = num_primary


        self.k = 3

        self.conv1 = spectral_norm(torch.nn.Conv2d(3, 4, (1, 1), padding=0))
        self.conv2 = spectral_norm(torch.nn.Conv2d(4, 4, (1, 1), padding=0))
        self.conv3 = spectral_norm(torch.nn.Conv2d(4, 4, (1, 1), padding=0))
        self.conv4 = spectral_norm(torch.nn.Conv2d(4, num_primary, (1, 1), padding=0))

        # self.conv1 = torch.nn.Conv2d(15, 8, (1, 1), padding=0)
        # self.conv2 = torch.nn.Conv2d(8, 8, (1, 1), padding=0)
        # self.conv3 = torch.nn.Conv2d(8, 8, (1, 1), padding=0)
        # self.conv4 = torch.nn.Conv2d(8, m, (1, 1), padding=0)

        torch.nn.init.xavier_normal_(self.conv1.weight)
        self.conv1.bias.data.fill_(1e-5)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        self.conv2.bias.data.fill_(1e-5)
        torch.nn.init.xavier_normal_(self.conv3.weight)
        self.conv3.bias.data.fill_(1e-5)
        torch.nn.init.xavier_normal_(self.conv4.weight)
        self.conv4.bias.data.fill_(1e-5)

    def forward(self, X):

        size = X.shape
        Y = X[:,0:3,:,:]

        temp = F.relu(self.conv1(X))
        temp = F.relu(self.conv2(temp))
        temp = F.relu(self.conv3(temp))
        res = self.conv4(temp)
        res[:,0:3,:,:] = res[:,0:3,:,:] + Y
        #res = self.cm(res)

        return res#, self.ref(res.view(3*size[0], self.k, 1, 1)).view(size[0], 3*400, 1, 1)


class weightConstraint(object):
    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, 'weight'):
            print("Entered")
            w = module.weight.data
            w = torch.clamp(w, min=0.0)
            module.weight.data = w