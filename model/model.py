import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBlock(nn.Module):
    def __init__(self):
        super(CNNBlock, self).__init__()
        # Define the first set of layers (x5 times)
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3,3,3), stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm3d(num_features=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)

        # Define the second set of layers (x1 time)
        self.conv2 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1,1,1), stride=1)
        self.batchnorm2 = nn.BatchNorm3d(num_features=1)

        # Define the third set of layers (x1 time)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.conv3 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1,1,1), stride=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Apply the first set of layers 5 times
        for _ in range(5):
            x = self.maxpool(self.relu(self.batchnorm1(self.conv1(x))))

        # Apply the second set of layers 1 time
        x = self.relu(self.batchnorm2(self.conv2(x)))

        # Apply the third set of layers 1 time
        x = self.conv3(self.dropout(self.avgpool(x)))

        # Softmax activation
        x = self.softmax(x)
        
        # Flatten the output for softmax
        x = torch.flatten(x, 1)
        
        return x

# You need to replace '?' with the appropriate numbers for in_channels and out_channels
# based on the architecture you are aiming to replicate.
