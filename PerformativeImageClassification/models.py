#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models


class PretrainedResNet(nn.Module):
    def __init__(self, num_classes):
        super(PretrainedResNet, self).__init__()

        # Load the pretrained ResNet18 model trained on CIFAR-10
        self.resnet = models.resnet18(pretrained=True)

        # Replace the last fully connected layer with a new one for the desired number of classes
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

class Twolayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        torch.manual_seed(args.seed)
        # self.lr = nn.Linear(args.input_dim, 1)
        self.cls = torch.nn.Sequential(
          nn.Linear(28*28, 14*14),
          nn.ReLU(),
          nn.Dropout(),
          nn.Linear(14*14, 10),
          nn.Softmax(dim=1)
        )
        for layer in self.cls:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight.data, 0, 0.01)
                nn.init.constant_(layer.bias.data, 0)

    def forward(self, input):
        input = input.flatten(start_dim=1)
        output = self.cls(input)
        return output
    
class Twolayer_private(nn.Module):
    def __init__(self, args):
        super().__init__()
        torch.manual_seed(args.seed)
        # self.lr = nn.Linear(args.input_dim, 1)
        
        self.conv = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.cls = torch.nn.Sequential(
          nn.Linear(28*28*3, 1000),
          nn.ReLU(),
          nn.Linear(1000, 100),
          nn.ReLU(),
          nn.Dropout(),
          nn.Linear(100, 10),
          nn.Softmax(dim=1)
        )
        for layer in self.cls:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight.data, 0, 0.01)
                nn.init.constant_(layer.bias.data, 0)

    def forward(self, input):
        input = input.reshape(-1,28,28).unsqueeze(1)
        output = self.conv(input).flatten(start_dim=1)
        output = self.cls(output)
        return output
    
# class Twolayer_private(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         torch.manual_seed(args.seed)
#         self.lr = nn.Linear(100, 10)
#         nn.init.normal_(self.lr.weight.data, 0, 0.01)
#         nn.init.constant_(self.lr.bias.data, 0)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, input):
#         output = self.lr(input)
#         output = self.softmax(output)
#         return output


class LogisticRegression(nn.Module):
    def __init__(self, args):
        super().__init__()
        torch.manual_seed(args.seed)
        self.lr = nn.Linear(20, 2)
        nn.init.normal_(self.lr.weight.data, 0, 0.01)
        nn.init.constant_(self.lr.bias.data, 0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        output = self.lr(input)
        output = self.softmax(output)
        return output

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(7*7*32, 26)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x.unsqueeze(1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return self.softmax(out)
