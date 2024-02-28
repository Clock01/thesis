import torch
import torch.nn as nn
import torch.nn.functional as F

#Rete adoperata per il dataset MNIST
class NetMNIST(nn.Module):
  def __init__(self):
      super(NetMNIST, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
      self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
      self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
      self.fc1 = nn.Linear(4*4*64, 256)
      self.fc2 = nn.Linear(256, 10)

  def forward(self, x):
      x = F.relu(self.conv1(x))
      x = F.relu(F.max_pool2d(self.conv2(x), 2))
      x = F.dropout(x, p=0.5, training=self.training)
      x = F.relu(F.max_pool2d(self.conv3(x), 2))
      x = F.dropout(x, p=0.5, training=self.training)
      x = x.view(-1,4*4*64)
      x = F.relu(self.fc1(x))
      x = F.dropout(x, training=self.training)
      x = self.fc2(x)
      return F.log_softmax(x, dim=1)

#Rete adoperata per il dataset FMNIST
class NetFMNIST(nn.Module):
    def __init__(self):
        super(NetFMNIST, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*7*7, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out

#Rete adoperata per il dataset Cifar10
class NetCifar10(nn.Module):
	def __init__(self):
		super(NetCifar10, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
		self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
		self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(64 * 4 * 4, 512)
		self.fc2 = nn.Linear(512, 10)
		self.dropout = nn.Dropout(0.5)
		
	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = self.pool(F.relu(self.conv3(x)))
		x = x.view(-1, 64 * 4 * 4)
		x = self.dropout(F.relu(self.fc1(x)))
		x = self.dropout(F.relu(self.fc2(x)))
		return x

