import os
import cv2
import torch
from model import CRNN
from train import ctc_decode
from config import *

def load_model(weights_path):
    model = CRNN(len(idx2char)).to(DEVICE)
    model.load_state_dict(torch.load(weights_path,map_location=DEVICE))
    model.eval()
    return model

def preprocess_image(img_path):
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))
    img = img/255.0
    img = torch.tensor(img,dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return img

def predict(model,img_tensor):
    with torch.no_grad():
        preds = model(img_tensor)
        preds_log_probs = preds.log_softmax(2)
        decoded = ctc_decode(preds_log_probs)
    return decoded[0]

def infer_file(model,img_path):
    img_tensor = preprocess_image(img_path)
    res = predict(model,img_tensor)
    print(f"{os.path.basename(img_path)} -> {res}")

def infer_folder(model,folder_path):
    for f in os.listdir(folder_path):
        if f.lower().endswith((".jpg",".png",".jpeg")):
            infer_file(model,os.path.join(folder_path,f))
