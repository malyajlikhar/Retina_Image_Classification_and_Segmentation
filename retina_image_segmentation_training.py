# retina_training.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import numpy as np
from retina_models import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2

# CONFIG
IMG_SIZE = (512, 512)
BATCH_SIZE = 2
NUM_EPOCHS = 30
PATIENCE = 5
SEGMENTATION_PATH = './processed_retina/segmentation'

os.makedirs('checkpoints', exist_ok=True)
os.makedirs('logs', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AUGMENTATION
transform = A.Compose([
    A.Resize(*IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])
# ----------------------------
# SEGMENTATION DATASET
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
        mask_path = os.path.join(self.mask_dir, file.replace('.png', '_mask.png'))

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        augmented = transform(image=image, mask=mask)
        return augmented['image'], augmented['mask'].long()

# ----------------------------
# SEGMENTATION TRAINING
# ----------------------------
def train_segmentation():
    model = UNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_ds = SegmentationDataset(f'{SEGMENTATION_PATH}/train.csv', f'{SEGMENTATION_PATH}/images', f'{SEGMENTATION_PATH}/masks')
    val_ds = SegmentationDataset(f'{SEGMENTATION_PATH}/val.csv', f'{SEGMENTATION_PATH}/images', f'{SEGMENTATION_PATH}/masks')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    best_val_loss = float('inf')
    patience_counter = 0
    log_file = open("logs/train_segmentation_log.txt", "w")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(imgs)
                loss = criterion(out, masks)
                val_loss += loss.item()

        log_line = (f"Epoch {epoch+1} | Train Loss: {train_loss / len(train_loader):.4f} | "
                    f"Val Loss: {val_loss / len(val_loader):.4f}\n")
        print(log_line.strip())
        log_file.write(log_line)
        log_file.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "checkpoints/best_segmentation.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping segmentation training.")
                break

    log_file.close()


# ----------------------------
# MAIN
# ----------------------------
if __name__ == '__main__':
    print("Training Segmentation Model...")
    train_segmentation()
