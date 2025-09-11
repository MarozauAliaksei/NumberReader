import time
import torch
import torch.nn as nn
import torch.optim as optim
import Levenshtein
from config import *

def cer(preds, targets):
    total_dist, total_len = 0, 0
    for p, t in zip(preds, targets):
        total_dist += Levenshtein.distance(p, t)
        total_len += len(t)
    return total_dist / max(1, total_len)

def train_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    start = time.time()

    for imgs, labels, label_lengths in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        label_lengths = label_lengths.to(DEVICE).long()

        preds = model(imgs)  # [B, W, C] -> CRNN
        preds = preds.permute(1, 0, 2)  # [W, B, C] для CTC

        input_lengths = torch.full(
            (preds.size(1),), fill_value=preds.size(0), dtype=torch.long, device=DEVICE
        )

        loss = criterion(preds, labels, input_lengths, label_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"[TRAIN] Epoch {epoch} | Loss: {avg_loss:.4f} | Time: {time.time()-start:.1f}s")
    return avg_loss

def validate(model, loader, criterion, epoch, beam_size=5):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for imgs, labels, label_lengths in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            label_lengths = label_lengths.to(DEVICE).long()

            preds = model(imgs)  # [B, W, C]
            preds_ctc = preds.permute(1, 0, 2)  # [W, B, C] для CTC

            batch_size = preds.size(0)
            seq_len = preds.size(1)
            input_lengths = torch.full(
                (batch_size,), fill_value=seq_len, dtype=torch.long, device=DEVICE
            )

            loss = criterion(preds_ctc, labels, input_lengths, label_lengths)
            total_loss += loss.item()

            decoded = ctc_beam_search_decode(preds, beam_size=100, target_len=None, stochastic=True)

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

    for p, t in list(zip(all_preds, all_targets))[:5]:
        print(f"   PRED: {p} | TRUE: {t}")

    return avg_loss, val_cer

import numpy as np

def ctc_beam_search_decode(preds, beam_size=100, target_len=None, stochastic=False):
    """
    preds: [B, W, C] лог-пробы (log_softmax)
    beam_size: количество лучших гипотез на каждом шаге
    target_len: максимальная длина последовательности (None = без ограничения)
    stochastic: если True — использовать вероятностный выбор символа для разнообразия
    """
    preds_log_probs = preds.cpu().numpy()
    B, W, C = preds_log_probs.shape
    results = []

    for b in range(B):
        seq = preds_log_probs[b]  # [W, C]
        beams = [(("", BLANK_IDX), 0.0)]  # (строка, последний символ), score

        for t in range(W):
            log_probs = seq[t]
            new_beams = []

            for (s, last), score in beams:
                if stochastic:
                    # вероятностный выбор символа
                    probs = np.exp(log_probs - log_probs.max())  # стабильный softmax
                    probs /= probs.sum()
                    sampled_c = np.random.choice(C, p=probs)
                    if sampled_c == BLANK_IDX:
                        new_beams.append(((s, BLANK_IDX), score + log_probs[BLANK_IDX]))
                    else:
                        new_s = s + (idx2char[sampled_c] if sampled_c != last else "")
                        new_beams.append(((new_s, sampled_c), score + log_probs[sampled_c]))
                else:
                    for c in range(C):
                        new_score = score + log_probs[c]
                        if c == BLANK_IDX:
                            new_beams.append(((s, BLANK_IDX), new_score))
                        else:
                            if c == last:
                                new_beams.append(((s, last), new_score))
                            else:
                                new_s = s + idx2char[c]
                                new_beams.append(((new_s, c), new_score))

            # оставляем только top-k
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]

        # выбираем лучшую гипотезу
        best_seq, _ = max(beams, key=lambda x: x[1])
        out_str = best_seq[0]

        # если задан target_len, обрезаем строку
        if target_len is not None and len(out_str) > target_len:
            out_str = out_str[:target_len]

        results.append(out_str)

    return results


def get_optimizer(model, lr=LR):
    return optim.Adam(model.parameters(), lr=lr)

def get_criterion():
    return nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
