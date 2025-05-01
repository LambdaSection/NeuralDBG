import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# Neural network model definition
class NeuralNetworkModel(nn.Module):
    def __init__(self):
        super(NeuralNetworkModel, self).__init__()
        self.layer3_dense = nn.Linear(in_features=64, out_features=64)
        self.layer4_dropout = nn.Dropout(p=0.5)
        self.layer5_output = nn.Sequential(nn.Linear(in_features=64, out_features=10), nn.Softmax(dim=1))

    # Forward pass
    def forward(self, x):
        x = self.layer3_dense(x)
        x = self.layer4_dropout(x)
        x = self.layer5_output(x)
        return x

# Model instantiation
model = NeuralNetworkModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Loss function
loss_fn = nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
