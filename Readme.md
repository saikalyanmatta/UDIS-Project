# 📦 Damaged vs Intact Packaging Detection (UDIS Project)

An **AIML + Digital Image Processing (DIP)** project to automatically detect whether a product package is **damaged or intact** using deep learning and explainable AI.

This project was developed as part of the **UDIS course**, with emphasis on:
- image preprocessing
- data augmentation
- CNN-based classification
- model explainability using Grad-CAM

---

## 🔍 Problem Statement

In logistics and e-commerce, identifying damaged packages manually is slow and error-prone.  
This project aims to **automatically classify package images** into:

- **Damaged**
- **Intact**

using a convolutional neural network trained on real-world package images.

---

## 🧠 Approach

### 1. Model Architecture
- **EfficientNet-B0** (pretrained on ImageNet)
- Final classifier layer modified for **binary classification**

### 2. Digital Image Processing (DIP)
Applied DIP-focused augmentations to improve robustness:
- Resizing to `224×224`
- Random rotations
- Horizontal flips
- Color jitter (brightness, contrast, saturation)
- Normalization

### 3. Training Strategy
- Loss: Cross-Entropy Loss
- Optimizer: Adam
- Best model saved using validation accuracy

---

## 📊 Evaluation Metrics

Model performance evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The model achieved **~94–95% validation accuracy** with balanced class-wise performance.

---

## 🔥 Explainable AI — Grad-CAM

To interpret model decisions, **Grad-CAM** was used to visualize regions contributing to predictions.

The heatmaps clearly highlight **damaged areas of the packaging**, increasing model transparency and trust.

<img width="950" height="315" alt="image" src="https://github.com/user-attachments/assets/533d8955-5b95-4818-9a6e-b412fccdf448" />


---

## 📁 Project Structure

UDIS-Project/
│
├── data/                      # Dataset (train / val split)
│   ├── train/
│   │   ├── damaged/
│   │   └── intact/
│   └── val/
│       ├── damaged/
│       └── intact/
│
├── models/
│   └── model.py               # DamageNet (EfficientNet-based CNN)
│
├── preprocess/
│   └── preprocessing.py       # Image preprocessing utilities
│
├── train.py                   # Training script
├── infer.py                   # Inference script
├── utils.py                   # Data loading & DIP augmentations
├── gradcam.py                 # Grad-CAM implementation
│
├── UDIS-Project.ipynb          # Training, evaluation & visualization notebook
├── .gitignore
└── README.md


