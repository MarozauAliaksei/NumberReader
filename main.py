import os
import random
import torch
from torch.utils.data import DataLoader
from train import train_epoch, validate, get_criterion, get_optimizer, ctc_beam_search_decode
from dataset import OCRDataset, collate_fn
from model import CRNN
from config import *
from Number_reader import DigitCNN  


TOTAL_EPOCHS = EPOCHS
PRETRAIN_EPOCHS = 0  
BATCH_SIZE = BATCH_SIZE
DEVICE = DEVICE
LR = LR

if __name__ == "__main__":

    train_ds = OCRDataset("data/train/images/augmented", augment=True)
    val_ds = OCRDataset("data/val/images", augment=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = CRNN(num_classes=11)
    model.to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if os.path.exists("digit_pretrain.pth"):
        print("⚡ Загружаем предобученные веса DigitCNN...")
        digit_cnn = DigitCNN(num_classes=10).to(DEVICE)
        digit_cnn.load_state_dict(torch.load("digit_pretrain.pth", map_location=DEVICE))
        digit_cnn.eval()

        model_dict = model.state_dict()
        cnn_dict = digit_cnn.state_dict()
        for k in cnn_dict.keys():
            if k in model_dict:
                model_dict[k] = cnn_dict[k]
        model.load_state_dict(model_dict)
        print("✅ Веса CNN скопированы в CRNN")


    criterion = get_criterion()
    optimizer = get_optimizer(model, LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_cer = float("inf")
    best_epoch = -1

    for epoch in range(1, TOTAL_EPOCHS + 1):
        train_epoch(model, train_loader, criterion, optimizer, epoch, beam_size=5, alpha=0.2, beta=0.2, gamma=0.5)

        val_loss, val_cer = validate(model, val_loader, criterion, epoch, beam_size=100)
        scheduler.step(val_loss)
        if val_cer < best_cer:
            best_cer = val_cer
            best_epoch = epoch
            torch.save(model.state_dict(), "crnn_best.pth")
            print(f"💾 Лучшая модель сохранена (Epoch {epoch}, CER={best_cer:.4f})")
        sample_idx = random.randint(0, len(train_ds) - 1)
        img_tensor, label_tensor, _ = train_ds[sample_idx]
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        model.eval()
        with torch.no_grad():
            preds = model(img_tensor) 
            decoded = ctc_beam_search_decode(preds.permute(1,0,2), beam_size=20)

        true_label = "".join([idx2char[i] for i in label_tensor.tolist()])
        print(f"\n📌 Пример после эпохи {epoch}:")
        print(f"   Истинная метка: {true_label}")
        print(f"   Предсказание:   {decoded[0]}")
        print("-" * 50)
        model.train()

    torch.save(model.state_dict(), "crnn_last.pth")
    print(f"✅ Обучение завершено. Лучшая модель: epoch {best_epoch}, CER={best_cer:.4f}")
