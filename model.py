import torch
import torch.nn as nn

class QuickDraw(nn.Module):
    def __init__(self,num_class):
        self.num_class = num_class
        super(QuickDraw, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3, padding=1),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=30, kernel_size=3, padding=1),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size=2)

        )

        self.flatten = nn.Flatten()

        self.dense1 = nn.Sequential(
            nn.Linear(7*7*30,7 * 7 * 15),
            nn.BatchNorm1d(7 * 7 * 15),
            nn.ReLU(inplace= True),
            nn.Dropout(p=0.5)

        )

        self.dense2 = nn.Sequential(
            nn.Linear(7 *7 * 15, 7 * 7),
            nn.BatchNorm1d(7 * 7),
            nn.ReLU(inplace= True),
            nn.Linear(7 * 7, num_class),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)


        x = self.flatten(x)

        x = self.dense1(x)
        x = self.dense2(x)

        return x