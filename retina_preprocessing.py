import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from random import sample
import matplotlib.patches as mpatches

# CONFIG
IMG_H, IMG_W = 512, 512  # maintains aspect ratio ~4288x2848
TARGET_COUNT_PER_CLASS = 1000
DATASET_PATH = '/home/turing/Downloads/B. Disease Grading'
SEGMENTATION_PATH = '/home/turing/Downloads/A. Segmentation'
PROCESSED_PATH = './processed_retina'

CLASSIFICATION_TEST_DIR = f'./{PROCESSED_PATH}/classification_test'
SEGMENTATION_TEST_DIR = f'./{PROCESSED_PATH}/segmentation_test'



os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(CLASSIFICATION_TEST_DIR, exist_ok=True)
os.makedirs(SEGMENTATION_TEST_DIR, exist_ok=True)


# AUGMENTATIONS
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Resize(IMG_H, IMG_W)
])

resize_only = A.Compose([
    A.Resize(IMG_H, IMG_W),
    A.Normalize(),
    ToTensorV2()
])

#Masked Maps
mask_map = {
        'MA': 1,
        'HE': 2,
        'EX': 3,
        'SE': 4,
        'OD': 5,
    }

code_to_dir = {
        'MA': '1. Microaneurysms',
        'HE': '2. Haemorrhages',
        'EX': '3. Hard Exudates',
        'SE': '4. Soft Exudates',
        'OD': '5. Optic Disc'
    }

# FUNCTION: Load Labels and Oversample
def balance_classification_data():
    train_csv = os.path.join(DATASET_PATH, '2. Groundtruths', 'a. IDRiD_Disease Grading_Training Labels.csv')
    df = pd.read_csv(train_csv)

    grouped = df.groupby(df.columns[1:3].tolist())  # Group by Retinopathy and Edema
    save_dir = os.path.join(PROCESSED_PATH, 'classification')
    os.makedirs(save_dir, exist_ok=True)
    metadata = []

    for (retino, edema), group in grouped:
        imgs = group['Image name'].values
        num_existing = len(imgs)
        num_needed = TARGET_COUNT_PER_CLASS - num_existing
        multiplier = max(0, num_needed // num_existing)

        for img_name in imgs:
            orig_path = os.path.join(DATASET_PATH, '1. Original Images', 'a. Training Set', img_name + '.jpg')
            image = cv2.imread(orig_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for i in range(multiplier + 1):
                aug_img = augment(image=image)['image']
                file_name = f'{img_name}_{i}_r{retino}_e{edema}.png'
                save_path = os.path.join(save_dir, file_name)
                img_np = aug_img.astype(np.uint8)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, img_np)
                metadata.append([file_name, retino, edema])

    pd.DataFrame(metadata, columns=['file', 'retino', 'edema']).to_csv(os.path.join(save_dir, 'labels.csv'), index=False)

# FUNCTION: Merge Segmentation Masks
def merge_segmentation_masks():
    train_imgs = sorted(glob(os.path.join(SEGMENTATION_PATH, '1. Original Images', 'a. Training Set', '*.jpg')))
    mask_root = os.path.join(SEGMENTATION_PATH, '2. All Segmentation Groundtruths', 'a. Training Set')
    save_img_dir = os.path.join(PROCESSED_PATH, 'segmentation', 'images')
    save_mask_dir = os.path.join(PROCESSED_PATH, 'segmentation', 'masks')
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_mask_dir, exist_ok=True)

    for path in tqdm(train_imgs):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_id = os.path.basename(path).split('.')[0]

        merged_mask = np.zeros(img.shape[:2], dtype=np.uint8)

        for code, label in mask_map.items():
            msk_dir = code_to_dir[code]
            msk_path = os.path.join(mask_root, msk_dir, f'{img_id}_{code}.tif')
            if os.path.exists(msk_path):
                msk = cv2.imread(msk_path)
                msk = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)
                red_mask = (msk[:, :, 0] == 255) & (msk[:, :, 1] == 0) & (msk[:, :, 2] == 0)
                merged_mask[red_mask] = label

        # Create 1000 total augmentations across all images
        for i in range(20):
            aug = augment(image=img, mask=merged_mask)
            aug_img, aug_mask = aug['image'], aug['mask']

            img_out = aug_img.astype(np.uint8)
            img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
            out_img = os.path.join(save_img_dir, f'{img_id}_{i}.png')
            cv2.imwrite(out_img, img_out)

            # Save merged mask
            mask_out = aug_mask.astype(np.uint8)
            out_mask = os.path.join(save_mask_dir, f'{img_id}_{i}_mask.png')
            cv2.imwrite(out_mask, mask_out)

# FUNCTION: Split Data
def split_train_val():
    # Split classification labels
    label_df = pd.read_csv(os.path.join(PROCESSED_PATH, 'classification', 'labels.csv'))
    train, val = train_test_split(label_df, test_size=0.2, stratify=label_df[['retino', 'edema']])
    train.to_csv(os.path.join(PROCESSED_PATH, 'classification', 'train.csv'), index=False)
    val.to_csv(os.path.join(PROCESSED_PATH, 'classification', 'val.csv'), index=False)

    # Split segmentation images
    img_paths = sorted(glob(os.path.join(PROCESSED_PATH, 'segmentation', 'images', '*.png')))
    base_names = [os.path.basename(p) for p in img_paths]
    train, val = train_test_split(base_names, test_size=0.2, random_state=42)
    pd.DataFrame({'file': train}).to_csv(os.path.join(PROCESSED_PATH, 'segmentation', 'train.csv'), index=False)
    pd.DataFrame({'file': val}).to_csv(os.path.join(PROCESSED_PATH, 'segmentation', 'val.csv'), index=False)

# FUNCTION: Prepare Test Classification Data
def prepare_test_classification():
    csv_path = f'{DATASET_PATH}/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv'
    img_dir = f'{DATASET_PATH}/1. Original Images/b. Testing Set/'
    df = pd.read_csv(csv_path)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        fname = row[df.columns[0]] + '.jpg'
        src = os.path.join(img_dir, fname)
        dst = os.path.join(CLASSIFICATION_TEST_DIR, fname)
        if os.path.exists(src):
            img = cv2.imread(src)
            img = cv2.resize(img, (IMG_H, IMG_W))
            cv2.imwrite(dst, img)
            rows.append({"file": fname, "retino": row[df.columns[1]], "edema": row[df.columns[2]]})

    pd.DataFrame(rows).to_csv(os.path.join(CLASSIFICATION_TEST_DIR, 'test.csv'), index=False)

# FUNCTION: Prepare Test Segmentation Data
def prepare_test_segmentation():
    SEGMENTATION_TEST_IMG_DIR = f'./{SEGMENTATION_TEST_DIR}/images'
    SEGMENTATION_TEST_MASK_DIR = f'./{PROCESSED_PATH}/segmentation_test/masks'
    os.makedirs(SEGMENTATION_TEST_IMG_DIR, exist_ok=True)
    os.makedirs(SEGMENTATION_TEST_MASK_DIR, exist_ok=True)
    input_img_dir = f'{SEGMENTATION_PATH}/1. Original Images/b. Testing Set/'
    input_mask_dir = f'{SEGMENTATION_PATH}/2. All Segmentation Groundtruths/b. Testing Set/'

    files = sorted([f for f in os.listdir(input_img_dir) if f.endswith('.jpg')])
    records = []

    for file in tqdm(files):
        img_id = file.split('.')[0]
        img_path = os.path.join(input_img_dir, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_H, IMG_W))
        cv2.imwrite(os.path.join(SEGMENTATION_TEST_IMG_DIR, file), img)

        merged_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)

        for code, label in mask_map.items():
            msk_dir = code_to_dir[code]
            mask_path = os.path.join(input_mask_dir, msk_dir, f'{img_id}_{code}.tif')
            if os.path.exists(mask_path):
                msk = cv2.imread(mask_path)
                msk = cv2.resize(msk, (IMG_H, IMG_W))
                msk = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)
                red_mask = (msk[:, :, 0] == 255) & (msk[:, :, 1] == 0) & (msk[:, :, 2] == 0)
                merged_mask[red_mask] = label

        out_mask_path = os.path.join(SEGMENTATION_TEST_MASK_DIR, file.replace('.jpg', '_mask.png'))
        cv2.imwrite(out_mask_path, merged_mask)
        records.append({"file": file})

    pd.DataFrame(records).to_csv(os.path.join(SEGMENTATION_TEST_DIR, 'test.csv'), index=False)

# MAIN
if __name__ == '__main__':
    print("Balancing classification dataset...")
    balance_classification_data()

    print("Creating merged segmentation masks...")
    merge_segmentation_masks()

    print("Splitting train/val sets...")
    split_train_val()

    print("Create classification test dataset...")
    prepare_test_classification()

    print("Create segmentation test dataset...")
    prepare_test_segmentation()
    
    print("Preprocessing Complete!")
