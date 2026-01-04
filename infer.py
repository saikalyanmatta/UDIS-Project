import torch
from PIL import Image
from torchvision import transforms
from models.model import DamageNet
import torch.nn.functional as F

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the model
model = DamageNet().to(device)
model.load_state_dict(
    torch.load("models/best_damage_model.pth", map_location=device)
)
model.eval()
 
# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probs = F.softmax(output, dim=1)
        pred = probs.argmax(1).item()
        confidence = probs.max().item()

    class_names = ["damaged", "intact"]
    return f"{class_names[pred]} ({confidence*100:.2f}%)"

print(predict("data/val/damaged/packagingboxesthataredamaged175.jpeg"))
print(predict("data/val/intact/packagingboxes73.jpeg"))
