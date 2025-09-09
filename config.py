import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Размеры картинок
IMG_HEIGHT = 56
IMG_WIDTH = 256

# Обучение
BATCH_SIZE = 16
EPOCHS = 150
LR = 1e-4

# Алфавит (только цифры)
ALPHABET = "0123456789"
BLANK_IDX = 0  # для CTC

# Маппинг символов
idx2char = ["-"] + list(ALPHABET)  # "-" = blank
char2idx = {c: i for i, c in enumerate(idx2char)}
