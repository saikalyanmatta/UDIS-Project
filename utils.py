import os
from torchvision import datasets, transforms
from torch.utils.data import random_split

# transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # Augmentations
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1,
        hue=0.02
    ),

    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


# Dataloader 
def get_dataloaders(data_dir, val_split=0.2):
    """
    Expects directory structure:
    data/
      train/
        damaged/
        intact/
      val/
        damaged/
        intact/
    """

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=val_transform
    )

    return train_dataset, val_dataset
