import time
import torch
import torch.nn as nn
import torch.optim as optim
import Levenshtein  # pip install python-Levenshtein
from config import *
# ------------------
# CER (Character Error Rate)
# ------------------
def cer(preds, targets):
    total_dist, total_len = 0, 0
    for p, t in zip(preds, targets):
        total_dist += Levenshtein.distance(p, t)
        total_len += len(t)
    return total_dist / max(1, total_len)

# ------------------
# Обучение одной эпохи
# ------------------
def train_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    start = time.time()
    for imgs, labels, label_lengths in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        label_lengths = label_lengths.to(DEVICE)

        preds = model(imgs)  # [W, B, C]
        preds_log_probs = preds.log_softmax(2)

        input_lengths = torch.full(
            size=(preds.size(1),), fill_value=preds.size(0), dtype=torch.long
        ).to(DEVICE)

        loss = criterion(preds_log_probs, labels, input_lengths, label_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"[TRAIN] Epoch {epoch} | Loss: {avg_loss:.4f} | Time: {time.time()-start:.1f}s")
    return avg_loss

# ------------------
# Валидация
# ------------------
def validate(model, loader, criterion, epoch):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, labels, label_lengths in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

    # ⚡️ Главное: label_lengths точно размером [batch_size] и типа long
            label_lengths = label_lengths.to(DEVICE).long()

            preds = model(imgs)               # [W, B, C]
            preds_log_probs = preds.log_softmax(2)

            batch_size = preds.size(1)
            seq_len = preds.size(0)

            input_lengths = torch.full(
                (batch_size,), fill_value=seq_len, dtype=torch.long, device=DEVICE
            )

            loss = criterion(preds_log_probs, labels, input_lengths, label_lengths)

            total_loss += loss.item()

            # декодируем предсказания
            decoded = ctc_decode(preds_log_probs)

            # превращаем labels обратно в текст
            labels_cpu = labels.cpu().numpy().tolist()
            label_lengths_cpu = label_lengths.cpu().numpy().tolist()
            idx = 0
            targets = []
            for l in label_lengths_cpu:
                word = "".join(idx2char[c] for c in labels_cpu[idx:idx+l])
                targets.append(word)
                idx += l

            all_preds.extend(decoded)
            all_targets.extend(targets)

    avg_loss = total_loss / len(loader)
    val_cer = cer(all_preds, all_targets)
    print(f"[VAL]   Epoch {epoch} | Loss: {avg_loss:.4f} | CER: {val_cer:.4f}")
    return avg_loss, val_cer

# ------------------
# Декодирование CTC
# ------------------
def ctc_decode(preds):
    preds = preds.permute(1, 0, 2)  # [B, W, C]
    pred_idxs = preds.argmax(2).cpu().numpy()
    results = []
    for idx_seq in pred_idxs:
        out_str = ""
        prev = BLANK_IDX
        for idx in idx_seq:
            if idx != prev and idx != BLANK_IDX:
                out_str += idx2char[idx]
            prev = idx
        results.append(out_str)
    return results

# ------------------
# Оптимизатор и критерий
# ------------------
def get_optimizer(model, lr=1e-3):
    return optim.Adam(model.parameters(), lr=lr)

def get_criterion():
    return nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
