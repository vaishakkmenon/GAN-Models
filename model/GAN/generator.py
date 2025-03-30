import torch.nn as nn

class Generator(nn.Module):
    # Latent_dim = Latent Dimension: Input noise vector
    # Img_shape = shape of image: 28x28
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        
        # First fully connected layer to convert from 100 to 128
        # Lifts noise vector to feature space representing data
        self.fc1 = nn.Linear(latent_dim, 128)
        
        # Second fully connected layer to convert from 128 to 256
        # Increases model capacity to learn complex features
        self.fc2 = nn.Linear(128, 256)
        
        # Normalizes the output of a layer so the next one learns better
        self.bn = nn.BatchNorm1d(256)
        
        # Final fully connected layer to convert from 256 to 768 (28 * 28)
        self.fc3 = nn.Linear(256, img_shape)
        
        # Keeps values positive and allows for complex patterns to be learned
        self.relu = nn.ReLU(inplace=True)
        
        # Scales all output values to be between -1 and 1
        self.tanh = nn.Tanh()
        
    def forward(self, z):
        # Pass noise into linear layer for learned features
        x = self.fc1(z)
        # Introduce non-linearity
        x = self.relu(x)
        
        # Increase learned features
        x = self.fc2(x)
        # Normalize output thus far; Training stabilization, better gradient flow, normalized input
        x = self.bn2(x)
        # Learn non-linear features from normalized data
        x = self.relu(x)
        
        # Final output layer to turn features into real-valued pixel data
        x = self.fc3(x)
        # Squash all output values to be between -1 and 1
        out = self.tanh(x)
        
        return out