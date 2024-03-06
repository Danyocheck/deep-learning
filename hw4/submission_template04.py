import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ВАШ КОД ЗДЕСЬ
        # определите слои сети

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5,5)) # 28
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2)) # 14
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3,3)) # 12
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2)) # 6

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(6 * 6 * 5, 100)
        self.fc2 = nn.Linear(100, 10)


    def forward(self, x):
        # размерность х ~ [64, 3, 32, 32]

        # ВАШ КОД ЗДЕСЬ
        # реализуйте forward pass сети

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def create_model():
    return ConvNet()



# ВАШ КОД ЗДЕСЬ
# объявите класс сверточной нейросети

# class ConvNet(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=(3,3)) # 30
#         self.pool1 = nn.MaxPool2d(kernel_size=(2,2)) # 15
#         self.conv2 = nn.Conv2d(in_channels=9, out_channels=6, kernel_size=(4,4)) # 12
#         self.pool2 = nn.MaxPool2d(kernel_size=(2,2)) # 6

#         self.flatten = nn.Flatten()

#         self.fc1 = nn.Linear(6 * 6 * 6, 128)
#         self.fc2 = nn.Linear(128, 10)


#     def forward(self, x):
#         # размерность х ~ [64, 3, 32, 32]

#         # ВАШ КОД ЗДЕСЬ
#         # реализуйте forward pass сети
#         x = F.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool2(x)

#         x = self.flatten(x)

#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
