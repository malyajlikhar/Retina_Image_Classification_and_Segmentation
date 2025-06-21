import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from retina_models import RetinaClassifier
import albumentations as A
from albumentations.pytorch import ToTensorV2

# CONFIG
IMG_SIZE = (512, 512)
CLASSIFICATION_TEST_PATH = './processed_retina/classification_test'
TEST_CLASSIFIER_CSV = f'{CLASSIFICATION_TEST_PATH}/test.csv'

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
# DATASET CLASSES
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
# CLASSIFICATION EVALUATION
# ----------------------------
def evaluate_classifier():
    model = RetinaClassifier().to(device)
    model.load_state_dict(torch.load("checkpoints/best_classifier.pt"))
    model.eval()

    test_df = pd.read_csv(TEST_CLASSIFIER_CSV)
    criterion = nn.CrossEntropyLoss()
    total, correct_retino, correct_edema, test_loss = 0, 0, 0, 0
    log_file = open("logs/test_classifier_log.txt", "w")

    y_true_retino, y_pred_retino = [], []
    y_true_edema, y_pred_edema = [], []

    with torch.no_grad():
        for i in range(len(test_df)):
            row = test_df.iloc[i]
            img_path = os.path.join(CLASSIFICATION_TEST_PATH, row['file'])
            image = np.array(Image.open(img_path).convert("RGB"))
            retino = torch.tensor(row['retino']).unsqueeze(0).to(device)
            edema = torch.tensor(row['edema']).unsqueeze(0).to(device)

            aug_img = transform(image=image)['image'].unsqueeze(0).to(device)
            out1, out2 = model(aug_img)
            loss = criterion(out1, retino) + criterion(out2, edema)
            test_loss += loss.item()

            pred1 = out1.argmax(1)
            pred2 = out2.argmax(1)
            correct_retino += (pred1 == retino).sum().item()
            correct_edema += (pred2 == edema).sum().item()
            total += 1

            y_true_retino.append(retino.item())
            y_pred_retino.append(pred1.item())
            y_true_edema.append(edema.item())
            y_pred_edema.append(pred2.item())

    acc_r = correct_retino / total
    acc_e = correct_edema / total
    log_line = (f"Test Loss: {test_loss / total:.4f} | Test Acc (Retino/Edema): {acc_r:.2%}/{acc_e:.2%}\n")
    print(log_line)
    log_file.write(log_line)

    # Classification report
    log_file.write("\nClassification Report for Retinopathy:\n")
    report_r = classification_report(y_true_retino, y_pred_retino)
    log_file.write(report_r + "\n")
    print(report_r)

    log_file.write("\nClassification Report for Edema:\n")
    report_e = classification_report(y_true_edema, y_pred_edema)
    log_file.write(report_e + "\n")
    print(report_e)

    # Confusion matrix plots
    cm_r = confusion_matrix(y_true_retino, y_pred_retino)
    cm_e = confusion_matrix(y_true_edema, y_pred_edema)

    disp_r = ConfusionMatrixDisplay(cm_r)
    disp_r.plot()
    plt.title("Retinopathy Confusion Matrix")
    plt.savefig("visuals/retino_confusion_matrix.png")
    plt.close()

    disp_e = ConfusionMatrixDisplay(cm_e)
    disp_e.plot()
    plt.title("Edema Confusion Matrix")
    plt.savefig("visuals/edema_confusion_matrix.png")
    plt.close()

    log_file.close()


if __name__ == "__main__":
    evaluate_classifier()
    print("Classification evaluation completed.")
    print("Results saved to logs/test_classifier_log.txt")