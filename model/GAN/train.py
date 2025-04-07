import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torchvision.utils import save_image

from generator import Generator
from discriminator import Discriminator
from utils.preprocessing import load_mnist_full

import matplotlib.pyplot as plt

# --- Config ---
latent_dim = 100
img_shape = 28 * 28
batch_size = 256
epochs = 100
save_dir = "generated-latest"
checkpoint_dir = "checkpoints-latest"

os.makedirs(save_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# --- Training Function ---
def train(rank, world_size):
    # Initializes the distributed training process
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    print(f"[INFO] Training on device: {device}")

    # Load and distribute dataset across multiple GPUs
    dataset = load_mnist_full()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    # Initialize models and wrap them with DDP
    G = DDP(Generator(latent_dim, img_shape).to(device), device_ids=[rank])
    D = DDP(Discriminator(img_shape, use_sigmoid=False).to(device), device_ids=[rank])

    # Binary cross entropy loss function used for both Discriminator and Generator
    # Optimized with Logits Loss for AMP to ensure stability
    criterion = nn.BCEWithLogitsLoss()

    # Set up optimizers for both models
    # Original = 0.0002; Learning Rate = 0.0003; Trying new value
    # Betas (0.5): Momentum term
    # Betas (0.999): Controls how quickly the optimizer adapts learning rates
    optimizer_G = optim.Adam(G.parameters(), lr=0.0003, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=0.0003, betas=(0.5, 0.999))

    # Cosine annealing learning rate schedulers
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=epochs, eta_min=1e-6)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=epochs, eta_min=1e-6)

    # AMP scalers for mixed precision training on RTX GPUs
    scaler_G = GradScaler('cuda')
    scaler_D = GradScaler('cuda')

    # Lists to store values for plotting at the end
    lr_history_G, lr_history_D, g_losses, d_losses = [], [], [], []

    for epoch in range(epochs):
        # Ensures shuffling across epochs for DistributedSampler
        sampler.set_epoch(epoch)  
        print(f"[INFO] Starting epoch {epoch} on rank {rank}")

        total_g_loss = 0.0
        total_d_loss = 0.0

        for batch_idx, (imgs, _) in enumerate(train_loader):
            imgs = imgs.view(-1, img_shape).to(device)  # Flatten and move images to device
            bs = imgs.size(0)
            
            # Real and fake labels
            valid = torch.ones(bs, 1, device=device)
            fake = torch.zeros(bs, 1, device=device)

            # ======================
            #  Train Discriminator
            # ======================
            
            # Random noise creation for generator to use
            z = torch.randn(bs, latent_dim, device=device)

            with autocast(device_type='cuda'):
                # Create fake images based on random noise
                # detach(): Stops backpropagation for generator so gradients are not updated
                fake_imgs = G(z).detach()
                # Teach discriminator real images
                real_loss = criterion(D(imgs), valid)
                # Teach discriminator fake images
                fake_loss = criterion(D(fake_imgs), fake)
                # Combine losses into overall loss; Balances contribtion of real and fake
                d_loss = (real_loss + fake_loss) / 2

            # Clear out any old gradients from the previous update
            # This prevents gradient accumulation across batches
            optimizer_D.zero_grad()

            # Scale the discriminator loss to amplify gradients for float16 precision
            # Helps prevent underflow and ensures gradient signal isn't lost
            scaler_D.scale(d_loss).backward()

            # Unscale the gradients and perform the optimizer step only if gradients are finite
            scaler_D.step(optimizer_D)

            # Adjust the scale factor for future steps based on gradient stability
            scaler_D.update()

            # ======================
            #  Train Generator
            # ======================
            
            # Random noise creation for generator to use
            z = torch.randn(bs, latent_dim, device=device)
            with autocast(device_type='cuda'):
                # Generate fake images
                gen_imgs = G(z)
                # Pass the generated images into the Discriminator, trying to fool Discriminator
                g_loss = criterion(D(gen_imgs), valid)

            # Clear out any old gradients from the previous update
            optimizer_G.zero_grad()

            # Scale the generator loss to amplify gradients for float16 precision
            # Helps prevent underflow and ensures gradient signal isn't lost
            scaler_G.scale(g_loss).backward()

            # Unscale the gradients and perform the optimizer step only if gradients are finite
            scaler_G.step(optimizer_G)

            # Adjust the scale factor for future steps based on gradient stability
            scaler_G.update()


            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

            if batch_idx % 100 == 0 and rank == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {batch_idx}/{len(train_loader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

        scheduler_G.step()
        scheduler_D.step()

        # ======================
        #  Save Output & Checkpoints
        # ======================
        if rank == 0:
            avg_g_loss = total_g_loss / len(train_loader)
            avg_d_loss = total_d_loss / len(train_loader)
            g_losses.append(avg_g_loss)
            d_losses.append(avg_d_loss)
            lr_history_G.append(scheduler_G.get_last_lr()[0])
            lr_history_D.append(scheduler_D.get_last_lr()[0])

            G.eval()
            with torch.no_grad():
                z = torch.randn(25, latent_dim, device=device)
                samples = G.module(z).view(-1, 1, 28, 28)
                save_image(samples, f"{save_dir}/epoch_{epoch}.png", nrow=5, normalize=True)
            G.train()

            torch.save(G.module.state_dict(), os.path.join(checkpoint_dir, f"generator_epoch_{epoch}.pth"))
            torch.save(D.module.state_dict(), os.path.join(checkpoint_dir, f"discriminator_epoch_{epoch}.pth"))

    # Clean up the process group
    print(f"[INFO] Finished training on rank {rank}, cleaning up...")
    dist.destroy_process_group()

    if rank == 0:
        plt.figure()
        plt.plot(g_losses, label='G Loss')
        plt.plot(d_losses, label='D Loss')
        plt.title('Generator and Discriminator Loss Per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig("loss_plot.png")

        plt.figure()
        plt.plot(lr_history_G, label='G LR')
        plt.plot(lr_history_D, label='D LR')
        plt.title('Cosine Annealing Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True)
        plt.savefig("lr_schedule_plot.png")

# --- Main Entry ---
if __name__ == "__main__":
    world_size = 4
    print("[INFO] Launching Training")
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
