import time
import torch
import torch.nn as nn
import torch.optim as optim
from config import DEVICE, BLANK_IDX, idx2char

def train_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    start = time.time()
    for imgs, labels, label_lengths in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs)
        preds_log_probs = preds.log_softmax(2)
        input_lengths = torch.full((preds.size(1),),preds.size(0),dtype=torch.long)
        loss = criterion(preds_log_probs, labels, input_lengths, label_lengths.to(DEVICE))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f} | Time: {time.time()-start:.1f}s")

def ctc_decode(preds):
    pred_idxs = preds.argmax(2).permute(1,0).cpu().numpy()
    results=[]
    for idx_seq in pred_idxs:
        s=""
        prev=-1
        for idx in idx_seq:
            if idx!=prev and idx!=BLANK_IDX:
                s+=idx2char[idx]
            prev=idx
        results.append(s)
    return results

def get_optimizer(model, lr):
    return optim.Adam(model.parameters(),lr=lr)

def get_criterion():
    return nn.CTCLoss(blank=BLANK_IDX,zero_infinity=True)
