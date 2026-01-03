import torch
from torch import nn, optim
from models.model import DamageNet
from utils import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = DamageNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# data loading
train_data, val_data = get_dataloaders("data")

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=16, shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    val_data, batch_size=16, shuffle=False
)

# accuracy function
def calculate_accuracy(loader, model):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            preds = outputs.argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

# training the model
epochs = 5
best_val_acc = 0.0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_acc = calculate_accuracy(train_loader, model)
    val_acc = calculate_accuracy(val_loader, model)

    print(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"Loss: {running_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    # ( Ssave the best model)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/best_damage_model.pth")
        print("✅ Best model saved")

print("Training complete.")
print("Best Validation Accuracy:", best_val_acc)
