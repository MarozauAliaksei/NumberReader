# -*- coding: utf-8 -*-
import os
import cv2
import torch
import torch.nn.functional as F
from model import CRNN  # твоя новая модель
from config import IMG_HEIGHT, IMG_WIDTH, idx2char

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Загрузка модели
# -----------------------------
def load_model(weights_path, num_classes=None):
    if num_classes is None:
        num_classes = len(idx2char)

    model = CRNN(num_classes=num_classes).to(DEVICE)
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


# -----------------------------
# Предобработка изображения
# -----------------------------
def preprocess_image(img_path, img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype('float32') / 255.0
    img = (img - 0.5) / 0.5  # нормализация
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return img_tensor


# -----------------------------
# Декодирование CTC
# -----------------------------
def ctc_decode(preds_log_probs):
    # preds_log_probs: [W,B,C] -> [B,W,C]
    preds = preds_log_probs.permute(1, 0, 2)
    preds = torch.argmax(preds, dim=-1).cpu().numpy()
    results = []
    for p in preds:
        prev = -1
        s = ""
        for i in p:
            if i != prev and i != 0:  # 0 - blank
                s += idx2char[i]
            prev = i
        results.append(s)
    return results


# -----------------------------
# Предсказание
# -----------------------------
def predict(model, img_tensor):
    with torch.no_grad():
        preds = model(img_tensor)  # [W,B,C]
        decoded = ctc_decode(preds)
    return decoded[0]


# -----------------------------
# Инференс на одном файле
# -----------------------------
def infer_file(model, img_path):
    img_tensor = preprocess_image(img_path)
    res = predict(model, img_tensor)
    print(f"{os.path.basename(img_path)} -> {res}")


# -----------------------------
# Инференс на папке
# -----------------------------
def infer_folder(model, folder_path):
    for f in os.listdir(folder_path):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            infer_file(model, os.path.join(folder_path, f))


# -----------------------------
# Пример запуска
# -----------------------------
if __name__ == "__main__":
    model = load_model("crnn_best.pth")  # путь к твоим весам
    infer_folder(model, "data/test/images")  # папка с изображениями
