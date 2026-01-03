from torchvision import datasets, transforms

def get_dataloaders(data_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # 🔥 FIX
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    train_data = datasets.ImageFolder(
        root=f"{data_dir}/train", transform=transform)

    val_data = datasets.ImageFolder(
        root=f"{data_dir}/val", transform=transform)

    return train_data, val_data
