# -*- coding: utf-8 -*-
import os
import cv2
import torch
from model import CRNN
from config import IMG_HEIGHT, IMG_WIDTH, idx2char
from train import ctc_beam_search_decode as ctc_decode

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(weights_path, num_classes=None):
    if num_classes is None:
        num_classes = len(idx2char)

    model = CRNN(num_classes=num_classes).to(DEVICE)
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def preprocess_image(img_path, img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype('float32') / 255.0
    img = (img - 0.5) / 0.5
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return img_tensor




def predict(model, img_tensor):
    with torch.no_grad():
        preds = model(img_tensor)  # [W,B,C]
        decoded = ctc_decode(preds)
    return decoded[0]


def infer_file(model, img_path):
    img_tensor = preprocess_image(img_path)
    res = predict(model, img_tensor)
    print(f"{os.path.basename(img_path)} -> {res}")


def infer_folder(model, folder_path):
    for f in os.listdir(folder_path):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            infer_file(model, os.path.join(folder_path, f))


if __name__ == "__main__":
    model = load_model("crnn_best.pth")
    infer_folder(model, "test")