import os
import torch
import random
from torch.utils.data import DataLoader, random_split
from train import ctc_beam_search_decode as ctc_decode
from dataset import *
from train import *
from model import *
from config import *
if __name__ == "__main__":
    train_dir = "data/train/images"
    val_dir = "data/val/images"

    # 📂 Загружаем весь датасет
    train_ds = OCRDataset(train_dir, augment=True)
    val_ds = OCRDataset(val_dir, augment=True)


    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # ⚙️ Модель, лосс, оптимизатор
    model = CRNN(len(idx2char)).to(DEVICE)
    criterion = get_criterion()
    optimizer = get_optimizer(model, LR)

    # 🎯 Лучший результат
    best_cer = float("inf")
    best_epoch = -1

    # 🚀 Обучение
    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, train_loader, criterion, optimizer, epoch)

        # 🔍 Валидация
        val_loss, val_cer = validate(model, val_loader, criterion, epoch)

        # ✅ Сохраняем лучшую модель
        if val_cer < best_cer:
            best_cer = val_cer
            best_epoch = epoch
            torch.save(model.state_dict(), "crnn_best.pth")
            print(f"💾 Лучшая модель сохранена (Epoch {epoch}, CER={val_cer:.4f})")

        # 🔹 Пример предсказания на случайной картинке из train
        sample_idx = random.randint(0, len(train_ds) - 1)
        img_tensor, label_tensor, _ = train_ds[sample_idx]
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        model.eval()
        with torch.no_grad():
            preds = model(img_tensor)
            preds_log_probs = preds.log_softmax(2)
            decoded = ctc_decode(preds_log_probs)

        true_label = "".join([idx2char[i] for i in label_tensor.tolist()])
        print(f"\n📌 Пример после эпохи {epoch}:")
        print(f"   Истинная метка: {true_label}")
        print(f"   Предсказание:   {decoded[0]}")
        print("-" * 50)
        model.train()

    # 💾 Сохраняем финальную модель
    torch.save(model.state_dict(), "crnn_last.pth")
    print(f"✅ Обучение завершено. Лучшая модель: epoch {best_epoch}, CER={best_cer:.4f}")
