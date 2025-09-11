import os
import torch
import random
from torch.utils.data import DataLoader
from train import ctc_beam_search_decode as ctc_decode
from dataset import *
from train import *
from model import *
from config import *
from Number_reader import DigitCNN  # Предобученная CNN

# -------------------------
# Настройки
# -------------------------
PRETRAIN_EPOCHS = 2 # сколько эпох обучаем только RNN, CNN заморожена
TOTAL_EPOCHS = EPOCHS  # общее число эпох обучения

if __name__ == "__main__":
    train_dir = "data/train/images/augmented"
    val_dir = "data/val/images"

    # -------------------------
    # Датасеты
    # -------------------------
    train_ds = OCRDataset(train_dir, augment=True)
    val_ds = OCRDataset(val_dir, augment=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # -------------------------
    # Модель
    # -------------------------
    model = CRNN(len(idx2char)).to(DEVICE)

    # ---------------------------------------
    # 🔥 Загрузка предобученных весов DigitCNN
    # ---------------------------------------
    if os.path.exists("digit_pretrain.pth"):
        print("⚡ Загружаем предобученные веса из digit_pretrain.pth...")
        digit_cnn = DigitCNN(num_classes=10).to(DEVICE)
        digit_cnn.load_state_dict(torch.load("digit_pretrain.pth", map_location=DEVICE))
        digit_cnn.eval()

        # Копируем сверточные слои по имени
        model_dict = model.state_dict()
        cnn_dict = digit_cnn.state_dict()
        mapping = {
            'conv1.weight': 'conv1.weight',
            'conv1.bias':   'conv1.bias',
            'conv2.weight': 'conv2.weight',
            'conv2.bias':   'conv2.bias',
            'conv3.weight': 'conv3.weight',
            'conv3.bias':   'conv3.bias',
            'conv4.weight': 'conv4.weight',
            'conv4.bias':   'conv4.bias',
        }
        pretrained_dict = {k: cnn_dict[v] for k,v in mapping.items() if v in cnn_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("✅ Веса CNN успешно скопированы в CRNN")

        # Заморозка CNN
        for name, param in model.named_parameters():
            if any(layer in name for layer in ['conv1', 'conv2', 'conv3', 'conv4', 'norm1', 'norm2', 'norm3', 'norm4']):
                param.requires_grad = False
        print("🔒 CNN заморожена на первые эпохи")
    else:
        print("⚠️ Файл digit_pretrain.pth не найден. Обучение с нуля.")

    # -------------------------
    # Лосс и оптимизатор
    # -------------------------
    criterion = get_criterion()
    optimizer = get_optimizer(model, LR)

    best_cer = float("inf")
    best_epoch = -1

    # -------------------------
    # Обучение
    # ------------------------
    CER = 1
    epoch = 1
    while CER > 1e-1:
        # Размораживаем CNN после PRETRAIN_EPOCHS
        if epoch == PRETRAIN_EPOCHS + 1:
            for name, param in model.named_parameters():
                if 'conv' in name:
                    param.requires_grad = True
            print("🔓 CNN разморожена, начинаем fine-tune всей модели")

        # --- Тренировка
        train_epoch(model, train_loader, criterion, optimizer, epoch)

        # --- Валидация
        val_loss, val_cer = validate(model, val_loader, criterion, epoch)

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
            preds = model(img_tensor)
            preds_log_probs = preds.log_softmax(2)
            decoded = ctc_decode(preds_log_probs)

        true_label = "".join([idx2char[i] for i in label_tensor.tolist()])
        print(f"\n📌 Пример после эпохи {epoch}:")
        print(f"   Истинная метка: {true_label}")
        print(f"   Предсказание:   {decoded[0]}")
        print("-" * 50)
        model.train()
        epoch += 1

    # -------------------------
    # Финальная модель
    # -------------------------
    torch.save(model.state_dict(), "crnn_last.pth")
    print(f"✅ Обучение завершено. Лучшая модель: epoch {best_epoch}, CER={best_cer:.4f}")
