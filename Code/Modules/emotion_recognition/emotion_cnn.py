import torch
import torch.nn as nn

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=2)
        
        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2a = nn.ReLU()
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2b = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(kernel_size=3, stride=2)
        
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3a = nn.ReLU()
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu3b = nn.ReLU()
        self.avgpool3 = nn.AvgPool2d(kernel_size=3, stride=2)
        
        # verify the output size of conv2 and conv3
        self.dummy_input = torch.randn(1, 1, 48, 48)
        self.dummy_output_size = self._get_conv_output_size(self.dummy_input)
        
        # update fc1 units based on feature map size
        self.fc1 = nn.Linear(self.dummy_output_size, 1024)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(1024, 1024)
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _get_conv_output_size(self, input_tensor):
        x = self.maxpool1(self.relu1(self.conv1(input_tensor)))
        x = self.relu2a(self.conv2a(x))
        x = self.relu2b(self.conv2b(x))
        x = self.avgpool2(x)
        x = self.relu3a(self.conv3a(x))
        x = self.relu3b(self.conv3b(x))
        x = self.avgpool3(x)
        return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.relu2a(self.conv2a(x))
        x = self.relu2b(self.conv2b(x))
        x = self.avgpool2(x)
        x = self.relu3a(self.conv3a(x))
        x = self.relu3b(self.conv3b(x))
        x = self.avgpool3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(self.relu_fc1(self.fc1(x)))
        x = self.dropout2(self.relu_fc2(self.fc2(x)))
        x = self.softmax(self.fc3(x))
        return x