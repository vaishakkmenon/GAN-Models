import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.module):
    # Latent_dim = Latent Dimension: Input noise vector
    # Img_shape = shape of image: 28x28
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.bn = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, img_shape)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        
    def forward(self, z):
        x = self.fc1(z)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        out = self.tanh(x)
        
        return out