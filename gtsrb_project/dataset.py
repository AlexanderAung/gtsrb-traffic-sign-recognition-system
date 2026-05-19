# Senior Engineer Note:
# This module handles loading and preprocessing GTSRB images into PyTorch tensors.

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# This function defines the exact image preprocessing shared by training and prediction.
def get_transforms():
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )


# This function loads the labeled GTSRB training folder using ImageFolder.
def load_dataset(dataset_root="traffic_signs"):
    train_folder = Path(dataset_root) / "Train"
    transform = get_transforms()

    return datasets.ImageFolder(root=train_folder, transform=transform)


# This function wraps the dataset in a DataLoader so training can read mini-batches.
def create_dataloader(dataset_root="traffic_signs", batch_size=32, shuffle=True):
    dataset = load_dataset(dataset_root)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
