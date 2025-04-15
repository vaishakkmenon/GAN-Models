#######################
# TO RUN
# python run_configs.py --config configs/config_000.yaml
#######################

import os
import sys
import yaml
import torch
from full_script import train

def run_with_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract config ID (e.g., 003 from config_003.yaml)
    config_id = os.path.splitext(os.path.basename(config_path))[0].split('_')[-1]

    # Create output directories for this config
    save_dir = f"generated/config_{config_id}"
    checkpoint_dir = f"checkpoints/config_{config_id}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Inject config values into full_script globals
    import full_script
    full_script.latent_dim = config.get('latent_dim', 100)
    full_script.batch_size = config.get('batch_size', 256)
    full_script.G_UPDATES_PER_D = config.get('G_UPDATES_PER_D', 1)
    full_script.optimizer_G_kwargs = dict(lr=config.get('G_lr', 0.0002), betas=(0.5, 0.999))
    full_script.optimizer_D_kwargs = dict(lr=config.get('D_lr', 0.0002), betas=(0.5, 0.999))
    full_script.scheduler_type = config.get('scheduler', 'cosine')

    # Set environment for DDP
    world_size = 4
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Run DDP training across all GPUs
    torch.multiprocessing.spawn(train, args=(world_size, save_dir, checkpoint_dir), nprocs=world_size, join=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run GAN training with a specific config.")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    run_with_config(args.config)