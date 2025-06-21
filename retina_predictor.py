# retina_predictor.py

import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
from retina_models import RetinaClassifier, UNet
import concurrent.futures

# CONFIG
IMG_SIZE = (512, 512)
CLASS_NAMES = ['Microaneurysms', 'Haemorrhages', 'Hard Exudates', 'Soft Exudates', 'Optic Disc']
CLASS_COLORS = [
    (255, 0, 0),      # Microaneurysms - Red
    (0, 255, 0),      # Haemorrhages - Green
    (255, 255, 0),    # Hard Exudates - Yellow
    (255, 0, 255),    # Soft Exudates - Magenta
    (0, 255, 255)     # Optic Disc - Cyan
]

transform = A.Compose([
    A.Resize(*IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# CLASSIFICATION PREDICTION
# ----------------------------
def predict_classification(image):
    model = RetinaClassifier().to(device)
    model.load_state_dict(torch.load("checkpoints/best_classifier.pt", map_location=device))
    model.eval()

    image_tensor = transform(image=image)['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        out1, out2 = model(image_tensor)
        pred1 = out1.argmax(1).item()
        pred2 = out2.argmax(1).item()

    retino_labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    edema_labels = ['No Risk', 'Mild', 'Severe']
    return retino_labels[pred1], edema_labels[pred2]

# ----------------------------
# SEGMENTATION PREDICTION
# ----------------------------
def predict_segmentation(image):
    model = UNet(n_classes=6).to(device)
    model.load_state_dict(torch.load("checkpoints/best_segmentation.pt", map_location=device))
    model.eval()

    image_tensor = transform(image=image)['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    overlay = np.array(image).copy()
    overlay = cv2.resize(overlay, IMG_SIZE)

    for label, color in enumerate(CLASS_COLORS, start=1):
        overlay[pred == label] = color

    return overlay

# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="Path to input image")
    args = parser.parse_args()

    image = np.array(Image.open(args.image).convert("RGB"))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        classification_future = executor.submit(predict_classification, image)
        segmentation_future = executor.submit(predict_segmentation, image)

        retino_label = classification_future.result()[0]
        edema_label = classification_future.result()[1]
        mask_overlay = segmentation_future.result()

    # Display classification results
    print(f"Retinopathy Grade: {retino_label}")
    print(f"Macular Edema Risk: {edema_label}")

    # Plot segmentation output
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_overlay)
    plt.axis('off')
    patches = [mpatches.Patch(color=np.array(c)/255, label=cls) for cls, c in zip(CLASS_NAMES, CLASS_COLORS)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Segmentation | Retino: {retino_label}, Edema: {edema_label}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
