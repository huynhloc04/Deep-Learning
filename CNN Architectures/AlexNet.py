
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=1000)

    def forward(self, x):
        x = self.relu(self.conv1(x))    #   (96x55x55)
        x = self.pool(x)                #   (96x27x27)
        x = self.relu(self.conv2(x))    #   (256x27x27)
        x = self.pool(x)                #   (256x13x13)
        x = self.relu(self.conv3(x))    #   (384x13x13))
        x = self.relu(self.conv4(x))    #   (384x13x13))
        x = self.relu(self.conv5(x))    #   (256x13x13))
        x = self.pool(x)                #   (256x6x6)
        x = x.reshape(x.shape[0], -1)   #   (9216, )
        x = self.relu(self.fc1(x))      #   (4096, )
        x = self.relu(self.fc2(x))      #   (4096, )
        x = self.relu(self.fc3(x))      #   (1000, )
        return x

if __name__ == "__main__":
    x = torch.randn(32, 3, 227, 227)
    model = AlexNet()
    print(model(x).shape)   #   Shape: (32x1000)