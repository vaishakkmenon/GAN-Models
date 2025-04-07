import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_shape, use_sigmoid=True):
        super(Discriminator, self).__init__()
        
        # First fully connected layer: takes 784 pixels -> 256 features
        self.fc1 = nn.Linear(img_shape, 256)
        # Second fully connected layer: takes 256 features -> 128 features
        self.fc2 = nn.Linear(256, 128)
        # Third fully connected layer: takes 128 features to 1 (real or fake)
        self.fc3 = nn.Linear(128, 1)
        
        # Nonlinear activation that avoids dead neurons.
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
        # Squashes output to [0, 1] to represent “realness"
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Pass the image through each layer, use leakyReLU for non-linear activation, 
        # and sigmoid for final output value
        # Sigmoid may be ignored based on loss function
        x = self.fc1(x)
        x = self.leaky_relu(x)

        x = self.fc2(x)
        x = self.leaky_relu(x)

        x = self.fc3(x)
        if self.use_sigmoid:
            x = self.sigmoid(x)

        return x