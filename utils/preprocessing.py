import os
import sys

from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class CelebADataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

def load_celeba(provided_path="img_align_celeba/", batch_size=64, num_workers=8):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_path = os.path.join(project_root, "data", provided_path)

    print(f"[INFO] Resolved dataset path: {data_path}")
    
    if project_root not in sys.path:
        sys.path.append(project_root)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The resolved path '{data_path}' does not exist.")

    train_dataset = datasets.MNIST(data_path, train=True, transform=transform)
    test_dataset = datasets.MNIST(data_path, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print("[INFO] Training Data Loaded")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print("[INFO] Testing Data Loaded")
    
    return train_loader, test_loader

def load_mnist(provided_path="mnist/", batch_size=64, num_workers=8):
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_path = os.path.join(project_root, "data", provided_path)

    print(f"[INFO] Resolved dataset path: {data_path}")
    
    if project_root not in sys.path:
        sys.path.append(project_root)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The resolved path '{data_path}' does not exist.")

    train_dataset = datasets.MNIST(data_path, train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print("[INFO] Training Data Loaded")
    
    return train_loader