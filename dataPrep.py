import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Function to apply Gaussian Noise
def add_gaussian_noise(image):
    """Adds Gaussian noise to the image."""
    noise_std = np.random.uniform(5, 30)  # Random noise intensity
    noisy_image = image + np.random.normal(0, noise_std, image.shape)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)  # Keep values in valid range

# Function to apply Random Rotation
def random_rotation(image):
    """Applies a random rotation within ±25 degrees."""
    rows, cols = image.shape[:2]
    angle = np.random.uniform(-25, 25)
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, rotation_matrix, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

# Define Gaussian Blur (Thickening) and Erosion (Thinning)
def random_gaussian_blur(image):
    """Applies Gaussian blur with random variance to make digits appear thicker."""
    sigma = np.random.uniform(0, 2)  # Random blur intensity
    return cv2.GaussianBlur(image, (5, 5), sigma)

def random_erosion(image):
    """Applies morphological erosion with a random kernel size to thin the digit."""
    kernel_size = 2
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

# Define augmentation pipeline with structured processing order
def apply_transformations(image):
    """Applies noise, rotation, and then thickening/thinning."""
    image = np.array(image.squeeze()) * 255  # Convert to uint8 scale (0-255)

    # Apply Gaussian Noise
    image = add_gaussian_noise(image)

    # Apply Random Rotation
    image = random_rotation(image)

    # Apply Random Thickening or Thinning
    if np.random.rand() < 1:
        image = random_gaussian_blur(image)  # Thickening
    else:
        image = random_erosion(image)  # Thinning

    return image

class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, augment_times=3):
        self.base_dataset = base_dataset
        self.augment_times = augment_times

    def __len__(self):
        return len(self.base_dataset) * (self.augment_times + 1)

    def __getitem__(self, idx):
        original_idx = idx % len(self.base_dataset)
        img, label = self.base_dataset[original_idx]

        transformed_img = apply_transformations(img)
        transformed_img = transforms.ToTensor()(transformed_img)
        

        label = torch.tensor(label, dtype=torch.long)
        real_fake_label = torch.tensor(1.0, dtype=torch.float32)

        return transformed_img, label, real_fake_label


def setup(SEED=42):
    # Set random seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load MNIST dataset
    emnist_dataset = datasets.EMNIST(
        root="./data", split="digits", train=True, download=True, transform=transforms.ToTensor()
    )

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split: 70% train, 15% val, 15% test
    total = len(emnist_dataset)
    val_size = int(0.15 * total)
    train_size = total - val_size
    print(f"Train size: {train_size}, Val size: {val_size}")
    train_base, val_base = random_split(emnist_dataset, [train_size, val_size])

    # Create augmented dataset for training
    train_data_aug = AugmentedDataset(train_base, augment_times=3)
    val_data = AugmentedDataset(val_base, augment_times=0)

    # Dataloader
    BATCH_SIZE = 1024
    train_loader = DataLoader(train_data_aug, num_workers=8, pin_memory=True, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, num_workers=8, pin_memory=True, batch_size=BATCH_SIZE, shuffle=False)


    return train_loader, val_loader, device

if __name__ == "__main__":
    # Example usage
    train_loader, val_loader, device = setup()