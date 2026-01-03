from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import torch

def generate_cam(model, img_tensor, img_path):
    target_layer = model.model.features[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])

    targets = [ClassifierOutputTarget(1)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]

    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0

    cam_image = show_cam_on_image(img, grayscale_cam)
    cv2.imwrite("cam_result.jpg", cam_image)
