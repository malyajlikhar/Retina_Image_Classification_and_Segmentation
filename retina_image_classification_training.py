# retina_training.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import numpy as np
from retina_models import RetinaClassifier
import albumentations as A
from albumentations.pytorch import ToTensorV2

# CONFIG
IMG_SIZE = (512, 512)
BATCH_SIZE = 2
NUM_EPOCHS = 30
PATIENCE = 5
CLASSIFICATION_PATH = './processed_retina/classification'

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
# CLASSIFICATION DATASET
# ----------------------------
class ClassificationDataset(Dataset):
    def __init__(self, csv_path, img_dir):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['file'])
        image = np.array(Image.open(img_path).convert("RGB"))
        image = transform(image=image)['image']
        return image, torch.tensor(row['retino']), torch.tensor(row['edema'])

# ----------------------------
# CLASSIFICATION TRAINING
# ----------------------------
def train_classifier():
    model = RetinaClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=model.l2_reg)

    train_ds = ClassificationDataset(f'{CLASSIFICATION_PATH}/train.csv', f'{CLASSIFICATION_PATH}')
    val_ds = ClassificationDataset(f'{CLASSIFICATION_PATH}/val.csv', f'{CLASSIFICATION_PATH}')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    best_val_loss = float('inf')
    patience_counter = 0
    log_file = open("logs/train_classifier_log.txt", "w")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, correct_retino, correct_edema, total = 0, 0, 0, 0
        for imgs, retino, edema in train_loader:
            imgs, retino, edema = imgs.to(device), retino.to(device), edema.to(device)
            optimizer.zero_grad()
            out1, out2 = model(imgs)
            loss = criterion(out1, retino) + criterion(out2, edema) + model.get_l2_loss()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct_retino += (out1.argmax(1) == retino).sum().item()
            correct_edema += (out2.argmax(1) == edema).sum().item()
            total += retino.size(0)

        model.eval()
        val_loss, val_correct_retino, val_correct_edema, val_total = 0, 0, 0, 0
        with torch.no_grad():
            for imgs, retino, edema in val_loader:
                imgs, retino, edema = imgs.to(device), retino.to(device), edema.to(device)
                out1, out2 = model(imgs)
                loss = criterion(out1, retino) + criterion(out2, edema)
                val_loss += loss.item()
                val_correct_retino += (out1.argmax(1) == retino).sum().item()
                val_correct_edema += (out2.argmax(1) == edema).sum().item()
                val_total += retino.size(0)

        train_acc_r = correct_retino / total
        train_acc_e = correct_edema / total
        val_acc_r = val_correct_retino / val_total
        val_acc_e = val_correct_edema / val_total

        log_line = (f"Epoch {epoch+1} | Train Loss: {train_loss / len(train_loader):.4f} | "
                    f"Train Acc (Retino/Edema): {train_acc_r:.2%}/{train_acc_e:.2%} | "
                    f"Val Loss: {val_loss / len(val_loader):.4f} | "
                    f"Val Acc (Retino/Edema): {val_acc_r:.2%}/{val_acc_e:.2%}\n")

        print(log_line.strip())
        log_file.write(log_line)
        log_file.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "checkpoints/best_classifier.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping classifier training.")
                break

    log_file.close()


# ----------------------------
# MAIN
# ----------------------------
if __name__ == '__main__':
    print("Training Classifier...")
    train_classifier()
