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


def validate(model, loader, criterion, epoch, beam_size=5):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, labels, label_lengths in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
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

            decoded = ctc_beam_search_decode(preds_log_probs, beam_size=beam_size)

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

    # 👀 Покажем первые 5 примеров
    for p, t in list(zip(all_preds, all_targets))[:5]:
        print(f"   PRED: {p} | TRUE: {t}")

    return avg_loss, val_cer


def ctc_beam_search_decode(preds_log_probs, beam_size=5, target_len=8):
    """
    Beam Search CTC decoder для цифр 0–9 + BLANK
    preds_log_probs: [W, B, C] (log softmax вероятности)
    """
    preds_log_probs = preds_log_probs.permute(1, 0, 2)  # [B, W, C]
    results = []
    for seq in preds_log_probs:  # [W, C]
        W, C = seq.shape
        beams = [(("", BLANK_IDX), 0.0)]  # (строка, последний_idx), лог-скор
        for t in range(W):
            log_probs = seq[t].cpu().numpy()
            new_beams = []
            for (s, last), score in beams:
                for c in range(C):
                    new_score = score + log_probs[c]
                    if c == BLANK_IDX:
                        new_beams.append(((s, last), new_score))
                    else:
                        if c == last:
                            # повтор символа → тот же s
                            new_beams.append(((s, c), new_score))
                        else:
                            new_s = s + idx2char[c]
                            new_beams.append(((new_s, c), new_score))
            # оставляем top beam_size
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]

        best_seq, _ = max(beams, key=lambda x: x[1])
        out_str = best_seq[0]

        # --- подгонка до target_len ---
        if len(out_str) > target_len:
            out_str = out_str[:target_len]
        elif len(out_str) < target_len:
            pad_char = out_str[-1] if out_str else "0"
            out_str = out_str.ljust(target_len, pad_char)

        results.append(out_str)
    return results


def get_optimizer(model, lr=1e-3):
    return optim.Adam(model.parameters(), lr=lr)


def get_criterion():
    return nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
