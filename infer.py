import torch
from PIL import Image
from torchvision import transforms
from models.model import DamageNet

model = DamageNet()
model.load_state_dict(torch.load("models/damage_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        pred = output.argmax(1).item()

    class_names = ['damaged', 'intact']
    return class_names[pred].capitalize()


print(predict("data/val/damaged/packagingboxesthataredamaged175.jpeg"))
print(predict("data/val/intact/packagingboxes73.jpeg"))
