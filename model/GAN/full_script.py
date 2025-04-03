import os
import sys
# Root path added for module access
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import StepLR

# Distributed Data Parallel Imports
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# AMP (Automatic Mixed Precision) imports for faster training on RTX GPUs
from torch.amp import autocast, GradScaler

# Config
latent_dim = 100
img_shape = 28 * 28
batch_size = 256
epochs = 100
warmup_epochs = 25
save_dir = "generated-latest"
os.makedirs(save_dir, exist_ok=True)

# Checkpoint directory
checkpoint_dir = "checkpoints-latest"
os.makedirs(checkpoint_dir, exist_ok=True)

def load_mnist_full(provided_path="mnist/", batch_size=64, num_workers=8):
    # Compose the transforms to normalize MNIST images to [-1, 1] range
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Go two directories up to find the root project directory
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_root = os.path.join(models_dir, "data")  # Ensures 'data/' is alongside 'models/'
    data_path = os.path.join(data_root, provided_path)

    os.makedirs(data_path, exist_ok=True)  # Ensure full directory path exists

    print(f"[INFO] Resolved dataset path: {data_path}")  

    if models_dir not in sys.path:
        sys.path.append(models_dir)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The resolved path '{data_path}' does not exist.")

    # Download both training and testing sets to combine
    train_dataset = datasets.MNIST(data_path, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(data_path, train=False, transform=transform, download=True)

    # Combine both into one dataset for more diversity
    full_dataset = ConcatDataset([train_dataset, test_dataset])
    print("[INFO] Returning Full Dataset for DDP")  

    return full_dataset

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
        x = self.bn(x)
        # Learn non-linear features from normalized data
        x = self.relu(x)

        # Final output layer to turn features into real-valued pixel data
        x = self.fc3(x)
        # Squash all output values to be between -1 and 1
        out = self.tanh(x)

        return out

class Discriminator(nn.Module):
    def __init__(self, img_shape):
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
        x = self.fc1(x)
        x = self.leaky_relu(x)

        x = self.fc2(x)
        x = self.leaky_relu(x)

        x = self.fc3(x)
        return x 

def train(rank, world_size):
    # Initializes the distributed training process
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    print(device)

    # Load and distribute dataset across multiple GPUs
    dataset = load_mnist_full()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    # Initialize models
    G = Generator(latent_dim, img_shape).to(device)
    D = Discriminator(img_shape).to(device)
    G = DDP(G, device_ids=[rank])
    D = DDP(D, device_ids=[rank])

    # Binary cross entropy loss function used for both Discriminator and Generator
    criterion = nn.BCEWithLogitsLoss()

    # Adam optimizers for Generator and Discriminator
    optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Learning Rate Scheduler for optimizers
    scheduler_G = StepLR(optimizer_G, step_size=10, gamma=0.5)
    scheduler_D = StepLR(optimizer_D, step_size=10, gamma=0.5)

    # AMP scalers for mixed precision training on RTX GPUs
    scaler_G = GradScaler('cuda')
    scaler_D = GradScaler('cuda')

    best_g_loss = float('inf')
    patience = 20
    patience_counter = 0

    # Begin training over specified number of epochs
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Ensures shuffling across epochs for DistributedSampler
        print(f"[INFO] Starting epoch {epoch} on rank {rank}")

        total_g_loss = 0.0
        total_d_loss = 0.0

        for batch_idx, (imgs, _) in enumerate(train_loader):
            imgs = imgs.view(-1, img_shape).to(device)        # Flatten and move images to correct device
            batch_size_curr = imgs.size(0)

            # Real and fake label vectors
            valid = torch.ones(batch_size_curr, 1, device=device)
            fake = torch.zeros(batch_size_curr, 1, device=device)

            # ======================
            #  Train Discriminator
            # ======================

            # Random noise creation for generator to use
            z = torch.randn(batch_size_curr, latent_dim, device=device)

            # Use AMP for forward pass
            with autocast(device_type='cuda'):
                fake_imgs = G(z).detach()
                real_loss = criterion(D(imgs), valid)
                fake_loss = criterion(D(fake_imgs), fake)
                d_loss = (real_loss + fake_loss) / 2

            # Optimize with AMP scaler
            optimizer_D.zero_grad()
            scaler_D.scale(d_loss).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()

            # ======================
            #  Train Generator
            # ======================

            z = torch.randn(batch_size_curr, latent_dim, device=device)
            with autocast(device_type='cuda'):
                gen_imgs = G(z)
                g_loss = criterion(D(gen_imgs), valid)

            optimizer_G.zero_grad()
            scaler_G.scale(g_loss).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()

            # Print training progress for monitoring
            if batch_idx % 100 == 0 and rank == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {batch_idx}/{len(train_loader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

        # Step LR schedulers
        scheduler_G.step()
        scheduler_D.step()

        # ======================
        #  Save Output & Checkpoints
        # ======================
        if rank == 0:
            avg_g_loss = total_g_loss / len(train_loader)
            avg_d_loss = total_d_loss / len(train_loader)
            lr_G = scheduler_G.get_last_lr()[0]
            lr_D = scheduler_D.get_last_lr()[0]

            print(f"[Epoch {epoch}/{epochs}] Avg D Loss: {avg_d_loss:.4f} | "
                f"Avg G Loss: {avg_g_loss:.4f} | LR_G: {lr_G:.6f} | LR_D: {lr_D:.6f}")
            print(f"[INFO] Saving checkpoint and samples for epoch {epoch}")

            G.eval()
            with torch.no_grad():
                z = torch.randn(25, latent_dim, device=device)
                samples = G.module(z).view(-1, 1, 28, 28)
                save_image(samples, f"{save_dir}/epoch_{epoch}.png", nrow=5, normalize=True)
            G.train()

            # Save checkpoint
            torch.save(G.module.state_dict(), os.path.join(checkpoint_dir, f"generator_epoch_{epoch}.pth"))
            torch.save(D.module.state_dict(), os.path.join(checkpoint_dir, f"discriminator_epoch_{epoch}.pth"))

            # Save best Generator model or update early stopping counter
            if avg_g_loss < best_g_loss:
                best_g_loss = avg_g_loss
                patience_counter = 0
                torch.save(G.module.state_dict(), os.path.join(checkpoint_dir, "best_generator.pth"))
                torch.save(D.module.state_dict(), os.path.join(checkpoint_dir, "best_discriminator.pth"))
                print(f"[INFO] New best model saved with G loss: {best_g_loss:.4f}")
            elif epoch >= warmup_epochs:
                patience_counter += 1
                print(f"[INFO] No improvement. Early stopping counter: {patience_counter}/{patience}")
                print(f"[INFO] Best G Loss: {best_g_loss}; Latest G Loss: {avg_g_loss}")
                if patience_counter >= patience:
                    print(f"[INFO] Early stopping triggered at epoch {epoch}")
                    break
            else:
                print(f"[INFO] Warm-up epoch {epoch} – skipping early stopping check.")

    # Clean up the process group
    print(f"[INFO] Finished training on rank {rank}, cleaning up...")
    dist.destroy_process_group()

# Launch distributed training using multiple processes (one per GPU)
if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    print("[INFO] Launching Training")  
    world_size = 4  # <-- MODIFIED to use 4 GPUs
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    print("[INFO] Training complete.")  