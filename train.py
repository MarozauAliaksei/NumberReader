import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CTCLoss

from Reader import OCRDataset, ocr_collate_fn
from RCNN_CTC import CRNN
from config import train_config as config


def train_batch(crnn, data, optimizer, criterion, device):
    crnn.train()  # ставим модель в режим обучения (включает dropout, batchnorm и т.д.)

    images, targets, target_lengths = [d.to(device) for d in data]
    # переносим тензоры на CPU или GPU

    logits = crnn(images)
    # прямой проход через модель (forward)
    # logits.shape = (seq_len, batch_size, num_classes)

    log_probs = torch.nn.functional.log_softmax(logits, dim=2)
    # CTC требует логарифмы вероятностей по классам

    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    target_lengths = torch.flatten(target_lengths)
    # длины последовательностей для CTC

    loss = criterion(log_probs, targets, input_lengths, target_lengths)
    # вычисляем CTC loss

    optimizer.zero_grad()  # обнуляем градиенты
    loss.backward()  # обратное распространение ошибки
    optimizer.step()  # обновляем веса модели

    return loss.item()  # возвращаем числовое значение лосса


def greedy_decode(logits, label2char):
    """
    Простая greedy-декодировка (без beam search).
    logits: (seq_len, batch, num_class)
    """
    preds = torch.argmax(logits, dim=2)  # (seq_len, batch)
    preds = preds.permute(1, 0)          # (batch, seq_len)

    results = []
    for seq in preds:
        string = []
        prev = -1
        for p in seq:
            p = p.item()
            if p != prev and p != 0:  # 0 = blank
                string.append(label2char[p])
            prev = p
        results.append("".join(string))
    return results


def cer(pred, target):
    """Character Error Rate"""
    import editdistance
    return editdistance.eval(pred, target) / max(1, len(target))


def evaluate(crnn, val_loader, criterion, device):
    crnn.eval()
    tot_loss, tot_count = 0, 0
    total_cer, total_wer, n_samples = 0, 0, 0

    with torch.no_grad():
        for images, targets, target_lengths in val_loader:
            images, targets, target_lengths = images.to(device), targets.to(device), target_lengths.to(device)

            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            batch_size = images.size(0)
            input_lengths = torch.LongTensor([logits.size(0)] * batch_size).to(device)
            target_lengths = torch.flatten(target_lengths)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            tot_loss += loss.item()
            tot_count += batch_size

            # decode
            preds = greedy_decode(logits, OCRDataset.LABEL2CHAR)
            # восстановим таргеты
            offset = 0
            for i, length in enumerate(target_lengths):
                true_seq = targets[offset:offset + length].cpu().numpy()
                gt_text = "".join([OCRDataset.LABEL2CHAR[idx] for idx in true_seq])
                offset += length

                total_cer += cer(preds[i], gt_text)
                n_samples += 1

    return tot_loss / tot_count, total_cer / n_samples, total_wer / n_samples


def main():
    epochs = config['epochs']
    train_batch_size = config['train_batch_size']
    val_batch_size = config['val_batch_size']

    lr = config['lr']
    show_interval = config['show_interval']
    valid_interval = config['valid_interval']
    save_interval = config['save_interval']
    cpu_workers = config['cpu_workers']
    reload_checkpoint = config['reload_checkpoint']

    img_width = config['img_width']
    img_height = config['img_height']
    data_dir = config['data_dir']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # train dataset
    train_dataset = OCRDataset(data_dir=data_dir, split='train',
                               img_height=img_height, img_width=img_width)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                              shuffle=True, num_workers=cpu_workers, collate_fn=ocr_collate_fn)

    # val dataset
    val_dataset = OCRDataset(data_dir=data_dir, split='val',
                             img_height=img_height, img_width=img_width)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size,
                            shuffle=False, num_workers=cpu_workers, collate_fn=ocr_collate_fn)

    num_class = len(OCRDataset.LABEL2CHAR) + 1
    crnn = CRNN(1, img_height, img_width, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])

    if reload_checkpoint:
        crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)

    optimizer = optim.RMSprop(crnn.parameters(), lr=lr)
    criterion = CTCLoss(reduction='sum').to(device)

    assert save_interval % valid_interval == 0 or valid_interval % save_interval == 0
    i = 1
    for epoch in range(1, epochs + 1):
        print(f'epoch: {epoch}')
        tot_train_loss = 0.
        tot_train_count = 0
        for train_data in train_loader:
            loss = train_batch(crnn, train_data, optimizer, criterion, device)
            train_size = train_data[0].size(0)

            tot_train_loss += loss
            tot_train_count += train_size
            if i % show_interval == 0:
                print('train_batch_loss[', i, ']: ', loss / train_size)

            if i % save_interval == 0:
                save_model_path = os.path.join(config["checkpoints_dir"], "crnn.pt")
                torch.save(crnn.state_dict(), save_model_path)
                print('save model at ', save_model_path)

            i += 1

        print('train_loss: ', tot_train_loss / tot_train_count)

        # 🔥 validation
        val_loss, val_cer, val_wer = evaluate(crnn, val_loader, criterion, device)
        print(f'val_loss: {val_loss:.4f}, CER: {val_cer:.4f}, WER: {val_wer:.4f}')


if __name__ == '__main__':
    main()
