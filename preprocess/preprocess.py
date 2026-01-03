import cv2
import numpy as np

def preprocess_image(img_path):
    img = cv2.imread(img_path)

    # Resize
    img = cv2.resize(img, (224, 224))

    # Convert to LAB for CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl,a,b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Edge enhancement
    edges = cv2.Canny(img, 50, 150)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Combine original + edges
    enhanced = cv2.addWeighted(img, 0.8, edges, 0.2, 0)

    return enhanced
