import os
import sys
# Root path added for module access
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.utils import save_image
from utils.preprocessing import load_mnist_full

from generator import Generator
from discriminator import Discriminator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

latent_dim = 100
img_shape = 28 * 28
epochs = 50
batch_size = 64
save_dir = "generated"

os.makedirs(save_dir, exist_ok=True)

train_data = load_mnist_full()

# Initialize Generator and Discriminator
G = Generator(latent_dim, img_shape).to(device)
D = Discriminator(img_shape).to(device)

# Binary Cross Entropy Loss
criterion = nn.BCELoss()

# Set up optimizers for both models
# Learning Rate = 0.0002; Commonly used in GANs
# Betas (0.5): Momentum term
# Betas (0.999): Controls how quickly the optimizer adapts learning rates
optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(epochs):
    for batch_idx, (imgs, _) in enumerate(train_data):
        imgs = imgs.view(-1, img_shape).to(device)
        batch_size = imgs.size(0)
        
        # Real and fake labels
        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)
        
        # ======================
        #  Train Discriminator
        # ======================
        # Random noise creation for generator to use
        z = torch.randn(batch_size, latent_dim, device=device)
        
        # Create fake images based on random noise
        # detach(): Stops backpropagation for generator so gradients are not updated
        fake_imgs = G(z).detach()

        # Teach discriminator real images
        real_loss = criterion(D(imgs), valid)
        
        # Teach discriminator fake images
        fake_loss = criterion(D(fake_imgs), fake)
        
        # Combine losses into overall loss; Balances contribtion of real and fake
        d_loss = (real_loss + fake_loss) / 2

        # Manually clear the gradients before the next backward pass
        # Prevents mixing gradients from previous batches
        optimizer_D.zero_grad()
        
        # Compute gradients of d_loss
        d_loss.backward()
        
        # Applies the weight updates using the Adam optimizer
        optimizer_D.step()
        
        # ======================
        #  Train Generator
        # ======================
        
        # Random noise creation for generator to use
        z = torch.randn(batch_size, latent_dim, device=device)
        
        # Generate fake images
        gen_imgs = G(z)

        # Pass the generated images into the Discriminator, trying to fool Discriminator
        g_loss = criterion(D(gen_imgs), valid)

        # Clears out any old gradients stored from the previous update
        optimizer_G.zero_grad()
        
        # Computes gradients of g_loss
        g_loss.backward()
        
        # Applies the gradient updates to the Generator using the Adam optimizer
        optimizer_G.step()

        # Print progress
        if batch_idx % 100 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {batch_idx}/{len(train_data)}] "
                f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    # Save sample images
    with torch.no_grad():
        z = torch.randn(25, latent_dim, device=device)
        samples = G(z).view(-1, 1, 28, 28)
        save_image(samples, f"{save_dir}/epoch_{epoch}.png", nrow=5, normalize=True)