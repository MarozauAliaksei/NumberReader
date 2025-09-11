import os
import torch
import random
from torch.utils.data import DataLoader
from train import ctc_beam_search_decode as ctc_decode, train_epoch, validate, get_criterion, get_optimizer
from dataset import OCRDataset, collate_fn
from model import CRNN  # или EfficientCRNN, если используем её
from config import *
from Number_reader import DigitCNN  # Предобученная CNN

# -------------------------
# Настройки
# -------------------------
PRETRAIN_EPOCHS = 0  # сразу финетюн всей модели
TOTAL_EPOCHS = EPOCHS

if __name__ == "__main__":
    train_dir = "data/train/images/augmented"
    val_dir = "data/val/images"

    # -------------------------
    # Датасеты
    # -------------------------
    train_ds = OCRDataset(train_dir, augment=True)
    val_ds = OCRDataset(val_dir, augment=True)

    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True,       # перемешиваем батчи
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        collate_fn=collate_fn
    )

    # -------------------------
    # Модель
    # -------------------------
    model = CRNN(in_channel=1, num_classes=11, cnn_input_height=32, use_gru=True)
    model.to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # -------------------------
    # Загрузка предобученной CNN
    # -------------------------
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

    # -------------------------
    # Лосс, оптимизатор и scheduler
    # -------------------------
    criterion = get_criterion()
    optimizer = get_optimizer(model, LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    best_cer = float("inf")
    best_epoch = -1

    # -------------------------
    # Обучение
    # -------------------------
    for epoch in range(1, TOTAL_EPOCHS + 1):
        # --- Тренировка
        train_epoch(model, train_loader, criterion, optimizer, epoch)

        # --- Валидация
        val_loss, val_cer = validate(model, val_loader, criterion, epoch)

        # --- Scheduler
        scheduler.step(val_loss)

        # --- Сохранение лучшей модели
        if val_cer < best_cer:
            best_cer = val_cer
            best_epoch = epoch
            torch.save(model.state_dict(), "crnn_best.pth")
            print(f"💾 Лучшая модель сохранена (Epoch {epoch}, CER={val_cer:.4f})")

        # --- Пример на случайной картинке
        sample_idx = random.randint(0, len(train_ds) - 1)
        img_tensor, label_tensor, _ = train_ds[sample_idx]
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        model.eval()
        with torch.no_grad():
            preds = model(img_tensor)  # [B, W, C]
            decoded = ctc_decode(preds)

        true_label = "".join([idx2char[i] for i in label_tensor.tolist()])
        print(f"\n📌 Пример после эпохи {epoch}:")
        print(f"   Истинная метка: {true_label}")
        print(f"   Предсказание:   {decoded[0]}")
        print("-" * 50)
        model.train()

    # -------------------------
    # Финальная модель
    # -------------------------
    torch.save(model.state_dict(), "crnn_last.pth")
    print(f"✅ Обучение завершено. Лучшая модель: epoch {best_epoch}, CER={best_cer:.4f}")
