import os
import torch
import random
from torch.utils.data import DataLoader
from dataset import OCRDataset, collate_fn
from model import CRNN
from train import train_epoch, get_optimizer, get_criterion, ctc_decode
from config import *
from infer import preprocess_image

if __name__=="__main__":
    train_dir="data/train/images"
    dataset = OCRDataset(train_dir)
    loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate_fn)

    model = CRNN(len(idx2char)).to(DEVICE)
    criterion = get_criterion()
    optimizer = get_optimizer(model,LR)

    for epoch in range(1,EPOCHS+1):
        train_epoch(model,loader,criterion,optimizer,epoch)

        # 🔹 Пример предсказания на случайной картинке
        sample_idx = random.randint(0,len(dataset)-1)
        img_tensor, label_tensor, _ = dataset[sample_idx]
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        model.eval()
        with torch.no_grad():
            preds = model(img_tensor)
            preds_log_probs = preds.log_softmax(2)
            decoded = ctc_decode(preds_log_probs)
        true_label = "".join([idx2char[i] for i in label_tensor.tolist()])
        print(f"Пример после эпохи {epoch}:")
        print(f"  Истинная метка: {true_label}")
        print(f"  Предсказание: {decoded[0]}")
        model.train()

    torch.save(model.state_dict(),"crnn_weights.pth")
    print("Модель сохранена")

