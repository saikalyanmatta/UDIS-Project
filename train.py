import torch
from torch import nn, optim
from models.model import DamageNet
from utils import get_dataloaders

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DamageNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_data, val_data = get_dataloaders("data")

train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

for epoch in range(5):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} completed")

torch.save(model.state_dict(), "models/damage_model.pth")

model.eval()
correct = 0
total = 0

val_loader = torch.utils.data.DataLoader(val_data, batch_size=16)

with torch.no_grad():
    for imgs, labels in val_loader:
        outputs = model(imgs.to(device))
        preds = outputs.argmax(1)
        correct += (preds.cpu() == labels).sum().item()
        total += labels.size(0)

print("Validation Accuracy:", correct / total)
