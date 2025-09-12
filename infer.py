# -*- coding: utf-8 -*-
import os
import cv2
import torch
import numpy as np
from model import CRNN
from config import IMG_HEIGHT, IMG_WIDTH, idx2char
from train import ctc_beam_search_decode as ctc_decode

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(weights_path, num_classes=None):
    """Загружает модель CRNN с заданными весами."""
    if num_classes is None:
        num_classes = len(idx2char)

    model = CRNN(num_classes=num_classes).to(DEVICE)
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def preprocess_image(img_path, img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
    """Чтение, ресайз и нормализация изображения."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение: {img_path}")

    img = cv2.resize(img, (img_width, img_height))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5  # нормализация в [-1, 1]
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1,1,H,W]
    return img_tensor


def predict(model, img_tensor, beam_size=20):
    """Предсказание текста с помощью CTC beam search."""
    with torch.no_grad():
        preds = model(img_tensor)  # [W,B,C]
        preds = preds.permute(1, 0, 2)  # [B,W,C] для декодирования
        decoded = ctc_decode(preds, beam_size=beam_size)
    return decoded[0]


def infer_file(model, img_path):
    """Инференс одного файла."""
    img_tensor = preprocess_image(img_path)
    result = predict(model, img_tensor)
    print(f"{os.path.basename(img_path)} -> {result}")


def infer_folder(model, folder_path, batch_size=8):
    """Инференс всех изображений в папке с батчами."""
    img_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    for i in range(0, len(img_files), batch_size):
        batch_files = img_files[i:i + batch_size]
        batch_tensors = [preprocess_image(f) for f in batch_files]
        batch_tensor = torch.cat(batch_tensors, dim=0)  # [B,1,H,W]

        with torch.no_grad():
            preds = model(batch_tensor)  # [W,B,C]
            preds = preds.permute(1, 0, 2)  # [B,W,C]
            decoded_batch = ctc_decode(preds, beam_size=20)

        for f, res in zip(batch_files, decoded_batch):
            print(f"{os.path.basename(f)} -> {res}")


if __name__ == "__main__":
    model = load_model("crnn_best.pth")
    test_folder = "test"
    infer_folder(model, test_folder)
