import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_HEIGHT = 56
IMG_WIDTH = 256

BATCH_SIZE = 32
EPOCHS = 400
LR = 5e-4

ALPHABET = "0123456789"
BLANK_IDX = 10  # для CTC

idx2char = ["-"] + list(ALPHABET)  # "-" = blank
char2idx = {c: i for i, c in enumerate(idx2char)}



