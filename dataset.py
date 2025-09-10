import os
import cv2
import torch
from torch.utils.data import Dataset
from config import *
import random
import numpy as np

# -----------------------------
# OCR Dataset (только 8 цифр) с аугментацией
# -----------------------------
class OCRDataset(Dataset):
    def __init__(self, root, augment=False):
        self.root = root
        self.augment = augment
        files = sorted([f for f in os.listdir(root) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        self.files = []
        self.labels = []
        for f in files:
            label = os.path.splitext(f)[0].split("_")[0]
            # ✅ фильтруем только 8-значные метки из допустимых символов
            if len(label) == 8 and all(c in char2idx for c in label):
                self.files.append(f)
                self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.files[idx])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # --- Ресайз по высоте с сохранением пропорций ---
        h, w = img.shape
        new_w = max(1, int(w * IMG_HEIGHT / h))
        img = cv2.resize(img, (new_w, IMG_HEIGHT))

        # --- Нормализация ---
        img = img.astype('float32') / 255.0
        img = (img - 0.5) / 0.5
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1,H,W]

        # --- Центрированный паддинг до IMG_WIDTH ---
        if new_w < IMG_WIDTH:
            pad_total = IMG_WIDTH - new_w
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            img_tensor = torch.nn.functional.pad(img_tensor, (pad_left, pad_right, 0, 0))
        elif new_w > IMG_WIDTH:
            img_tensor = img_tensor[:, :, :IMG_WIDTH]

        # --- Метка (ровно 8 цифр) ---
        label = torch.tensor([char2idx[c] for c in self.labels[idx]], dtype=torch.long)
        return img_tensor, label, len(label)

def collate_fn(batch):
    imgs, labels, label_lengths = zip(*batch)
    imgs = torch.stack(imgs)                   # [B,1,H,W]
    labels = torch.cat(labels)                 # [sum(label_lengths)]
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)  # [B]
    return imgs, labels, label_lengths
