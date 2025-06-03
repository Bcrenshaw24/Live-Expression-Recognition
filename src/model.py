import torch 
import torchvision 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


class NeuralNetwork(nn.Module): 
    def __init__(self):
        super().__init__() 
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) 
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2) 
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128*12*12, 256) 
        self.fc2 = nn.Linear(256, 64) 
        self.fc3 = nn.Linear(64, 7) 
    def forward(self, x): 
        x = F.relu(self.conv1(x)) 
        x = F.relu(self.conv2(x)) 
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.flatten(start_dim=1) 
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x) 
        return logits 
     