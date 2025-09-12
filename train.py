import time
import torch
import torch.nn as nn
import torch.optim as optim
import Levenshtein
import numpy as np
from config import *

# ---------------------------
# CER для отчета
# ---------------------------
def cer(preds, targets):
    total_dist, total_len = 0, 0
    for p, t in zip(preds, targets):
        total_dist += Levenshtein.distance(p, t)
        total_len += len(t)
    return total_dist / max(1, total_len)

# ---------------------------
# Доп. штрафы и бонусы
# ---------------------------
def length_penalty_exp(preds_strs, targets, base_weight=0.5):
    total = 0.0
    for p, t in zip(preds_strs, targets):
        diff = abs(len(p) - len(t))
        total += np.exp(diff) - 1
    return base_weight * total / len(targets)

def digit_set_penalty(preds_strs, targets, weight=1.0):
    total = 0.0
    for p, t in zip(preds_strs, targets):
        set_p, set_t = set(p), set(t)
        diff = len(set_t - set_p) + len(set_p - set_t)
        total += diff * weight
    return total / len(targets)

def bigram_reward(preds_strs, targets, weight=0.3):
    total = 0.0
    for p, t in zip(preds_strs, targets):
        min_len = min(len(p), len(t))
        for i in range(min_len - 1):
            if p[i] == t[i] and p[i+1] == t[i+1]:
                total -= weight
    return total / len(targets)

def repeated_digit_penalty(preds_strs, weight=0.2):
    total = 0.0
    for p in preds_strs:
        for i in range(1, len(p)):
            if p[i] == p[i-1]:
                total += weight
    return total / len(preds_strs)

def edge_digit_penalty(preds_strs, targets, weight=0.8):
    total = 0.0
    for p, t in zip(preds_strs, targets):
        if len(p) > 0 and p[0] != t[0]:
            total += weight
        if len(p) > 1 and p[-1] != t[-1]:
            total += weight
    return total / len(targets)

def positional_reward(preds_strs, targets, weight=0.3):
    total = 0.0
    for p, t in zip(preds_strs, targets):
        min_len = min(len(p), len(t))
        for i in range(1, min_len-1):
            if p[i] != t[i]:
                total += weight
    return total / len(targets)

# ---------------------------
# Beam search decoding
# ---------------------------
def ctc_beam_search_decode(preds, beam_size=100, target_len=None, stochastic=False):
    preds_log_probs = preds.detach().cpu().numpy()
    B, W, C = preds_log_probs.shape
    results = []

    for b in range(B):
        seq = preds_log_probs[b]
        beams = [(("", BLANK_IDX), 0.0)]

        for t in range(W):
            log_probs = seq[t]
            new_beams = []

            for (s, last), score in beams:
                if stochastic:
                    probs = np.exp(log_probs - log_probs.max())
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

            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]

        best_seq, _ = max(beams, key=lambda x: x[1])
        out_str = best_seq[0]
        if target_len is not None and len(out_str) > target_len:
            out_str = out_str[:target_len]
        results.append(out_str)

    return results

# ---------------------------
# Тренировка одной эпохи
# ---------------------------
def train_epoch(model, loader, criterion, optimizer, epoch,
                beam_size=200, alpha=0.2, beta=1.2, gamma=0.6,
                delta=0.05, eta=0.6, zeta=1.5, length_weight=0.2):
    model.train()
    total_loss = 0
    start = time.time()

    for imgs, labels, label_lengths in loader:
        imgs = imgs.to(DEVICE)
        label_lengths = label_lengths.to(DEVICE).long()

        if labels.dim() > 1:
            labels_1d = torch.cat([labels[i, :label_lengths[i]] for i in range(labels.size(0))])
        else:
            labels_1d = labels.to(DEVICE)

        preds = model(imgs)  # [W,B,C]
        preds_dec = preds.permute(1,0,2)  # [B,W,C]
        batch_size, seq_len, num_classes = preds_dec.shape

        preds_ctc = preds
        input_lengths = torch.full((batch_size,), fill_value=seq_len, dtype=torch.long, device=DEVICE)

        # ---------- CTC loss ----------
        ctc_loss = criterion(preds_ctc, labels_1d, input_lengths, label_lengths)

        # ---------- Beam Search декодирование ----------
        decoded = ctc_beam_search_decode(preds_dec, beam_size=beam_size)
        targets = []
        labels_cpu = labels.cpu().numpy().tolist()
        label_lengths_cpu = label_lengths.cpu().numpy().tolist()
        idx = 0
        for l in label_lengths_cpu:
            word = "".join(idx2char[c] for c in labels_cpu[idx:idx+l])
            targets.append(word)
            idx += l

        # ---------- комбинированный штраф/бонус ----------
        extra_loss = (
            alpha * positional_reward(decoded, targets) +
            beta * edge_digit_penalty(decoded, targets) +
            delta * repeated_digit_penalty(decoded) +
            zeta * digit_set_penalty(decoded, targets, weight=zeta) +
            length_weight * length_penalty_exp(decoded, targets, base_weight=length_weight) -
            eta * bigram_reward(decoded, targets, weight=eta)
        )

        loss = ctc_loss + extra_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"[TRAIN] Epoch {epoch} | Loss: {avg_loss:.4f} | Time: {time.time()-start:.1f}s")
    return avg_loss

# ---------------------------
# Валидация
# ---------------------------
def validate(model, loader, criterion, epoch, beam_size=100):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for imgs, labels, label_lengths in loader:
            imgs = imgs.to(DEVICE)
            label_lengths = label_lengths.to(DEVICE).long()

            if labels.dim() > 1:
                labels_1d = torch.cat([labels[i, :label_lengths[i]] for i in range(labels.size(0))])
            else:
                labels_1d = labels.to(DEVICE)

            preds = model(imgs)
            preds_ctc = preds

            batch_size, seq_len, num_classes = preds.permute(1,0,2).shape
            input_lengths = torch.full((batch_size,), fill_value=seq_len, dtype=torch.long, device=DEVICE)

            loss = criterion(preds_ctc, labels_1d, input_lengths, label_lengths)
            total_loss += loss.item()

            decoded = ctc_beam_search_decode(preds.permute(1,0,2), beam_size=beam_size, stochastic=True)

            targets = []
            labels_cpu = labels.cpu().numpy().tolist()
            label_lengths_cpu = label_lengths.cpu().numpy().tolist()
            idx = 0
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

# ---------------------------
# Оптимизатор и критерий
# ---------------------------
def get_optimizer(model, lr=LR):
    return optim.Adam(model.parameters(), lr=lr)

def get_criterion():
    return nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
