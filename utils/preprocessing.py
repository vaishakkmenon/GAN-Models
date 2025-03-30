import os
import sys

from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset


def resolve_data_path(provided_path):
    """
    Resolves the absolute path to the dataset, ensuring 'data/' is alongside 'models/' directory.
    """
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(models_dir, "data", provided_path)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    print(f"[INFO] Resolved dataset path: {data_path}")
    return data_path


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
        transforms.CenterCrop(178),
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_path = resolve_data_path(provided_path)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The resolved path '{data_path}' does not exist. "
                                f"Please place the CelebA dataset there.")

    train_data = CelebADataset(data_path, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print("[INFO] CelebA Training Data Loaded")

    return train_loader


def load_mnist(provided_path="mnist/", batch_size=64, num_workers=8):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data_path = resolve_data_path(provided_path)

    train_dataset = datasets.MNIST(data_path, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(data_path, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print("[INFO] MNIST Training Data Loaded")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print("[INFO] MNIST Testing Data Loaded")

    return train_loader, test_loader


def load_mnist_full(provided_path="mnist/", batch_size=64, num_workers=8):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data_path = resolve_data_path(provided_path)

    train_dataset = datasets.MNIST(data_path, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(data_path, train=False, transform=transform, download=True)

    full_dataset = ConcatDataset([train_dataset, test_dataset])
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    print("[INFO] MNIST Combined Train + Test Data Loaded")

    return full_loader


if __name__ == "__main__":
    # Example usage:
    load_mnist_full()  # You can replace this with load_mnist() or load_celeba()