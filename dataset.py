import os
import cv2
import torch
from torch.utils.data import Dataset
from config import *

# -----------------------------
# OCR Dataset (с центрированным паддингом)
# -----------------------------
class OCRDataset(Dataset):
    def __init__(self, root):
        self.root = root
        files = sorted([f for f in os.listdir(root) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        self.files = []
        self.labels = []
        for f in files:
            label = os.path.splitext(f)[0].split("_")[0]
            if all(c in char2idx for c in label):
                self.files.append(f)
                self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.files[idx])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # --- Ресайз по высоте с сохранением соотношения сторон ---
        h, w = img.shape
        new_w = max(1, int(w * IMG_HEIGHT / h))  # защита от нуля
        img = cv2.resize(img, (new_w, IMG_HEIGHT))

        # --- Нормализация (без бинаризации) ---
        img = img.astype('float32') / 255.0
        img = (img - 0.5) / 0.5

        # --- В Tensor ---
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1,H,W]

        # --- Центрированный паддинг по ширине ---
        if new_w < IMG_WIDTH:
            pad_total = IMG_WIDTH - new_w
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            img_tensor = torch.nn.functional.pad(img_tensor, (pad_left, pad_right, 0, 0))
        elif new_w > IMG_WIDTH:
            img_tensor = img_tensor[:, :, :IMG_WIDTH]

        # --- Массив меток ---
        label = torch.tensor([char2idx[c] for c in self.labels[idx]], dtype=torch.long)
        return img_tensor, label, len(label)


# -----------------------------
# Collate function для DataLoader
# -----------------------------
def collate_fn(batch):
    imgs, labels, label_lengths = zip(*batch)
    imgs = torch.stack(imgs)                   # [B,1,H,W]
    labels = torch.cat(labels)                 # [sum(label_lengths)]
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)  # [B]
    return imgs, labels, label_lengths
