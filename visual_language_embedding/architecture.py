import torch.nn as nn
import torch.nn.functional as F


class VLE_01(nn.Module):
    def __init__(self, num_clusters):
        super(VLE_01, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 32, 3)
        self.conv1_2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2_1 = nn.Conv2d(64, 64, 3)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3_1 = nn.Conv2d(64, 64, 3)
        self.conv3_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_clusters)

    def forward(self, x):
        x = F.relu(self.conv1_1(x)) # 70 x 70 -> 68 x 68
        x = F.relu(self.conv1_2(x)) # 68 x 68 -> 68 x 68
        x = self.pool1(x) # 68 x 68 -> 34 x 34

        x = F.relu(self.conv2_1(x))  # 34 x 34 -> 32 x 32
        x = F.relu(self.conv2_2(x))  # 32 x 32 -> 32 x 32
        x = self.pool2(x)  # 32 x 32 -> 16 x 16

        x = F.relu(self.conv3_1(x))  # 16 x 16 -> 14 x 14
        x = F.relu(self.conv3_2(x))  # 14 x 14 -> 14 x 14
        x = self.pool3(x)  # 14 x 14 -> 7 x 7

        x = x.view(-1, 7 * 7 * 64)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
