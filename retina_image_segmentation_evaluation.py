import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from retina_models import RetinaClassifier, UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2

# CONFIG
IMG_SIZE = (512, 512)
SEGMENTATION_TEST_PATH = './processed_retina/segmentation_test'
TEST_SEGMENTATION_CSV = f'{SEGMENTATION_TEST_PATH}/test.csv'


os.makedirs('logs', exist_ok=True)
os.makedirs('visuals', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AUGMENTATION
transform = A.Compose([
    A.Resize(*IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])

# ----------------------------
# DATASET CLASSES
# ----------------------------
class SegmentationDataset(Dataset):
    def __init__(self, csv_path, img_dir, mask_dir):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file = self.df.iloc[idx]['file']
        img_path = os.path.join(self.img_dir, file)
        mask_path = os.path.join(self.mask_dir, file.replace('.jpg', '_mask.png'))

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        augmented = transform(image=image, mask=mask)
        return augmented['image'], augmented['mask'].long()

# ----------------------------
# SEGMENTATION EVALUATION
# ----------------------------
def evaluate_segmentation():
    model = UNet().to(device)
    model.load_state_dict(torch.load("checkpoints/best_segmentation.pt"))
    model.eval()

    test_ds = SegmentationDataset(TEST_SEGMENTATION_CSV, f'{SEGMENTATION_TEST_PATH}/images', f'{SEGMENTATION_TEST_PATH}/masks')
    test_loader = DataLoader(test_ds, batch_size=1)

    log_file = open("logs/test_segmentation_log.txt", "w")
    iou_scores = []

    with torch.no_grad():
        for i, (img, mask) in enumerate(test_loader):
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            pred_cls = torch.argmax(pred, dim=1)

            iou = jaccard_score(mask.view(-1).cpu(), pred_cls.view(-1).cpu(), average='macro', zero_division=0)
            iou_scores.append(iou)

            # Visualization
            vis_img = img[0].permute(1, 2, 0).cpu().numpy()
            vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min())
            pred_mask = pred_cls[0].cpu().numpy()
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1); plt.imshow(vis_img); plt.title('Original')
            plt.subplot(1, 3, 2); plt.imshow(mask[0].cpu()); plt.title('Ground Truth')
            plt.subplot(1, 3, 3); plt.imshow(pred_mask); plt.title('Prediction')
            plt.savefig(f'visuals/test_seg_{i}.png')
            plt.close()

    avg_iou = np.mean(iou_scores)
    log_line = f"Avg IoU on test set: {avg_iou:.4f}\n"
    print(log_line)
    log_file.write(log_line)
    log_file.close()


if __name__ == "__main__":
    print("Evaluating segmentation model...")
    evaluate_segmentation()
    print("Evaluation complete. Check logs and visuals directory for results.")