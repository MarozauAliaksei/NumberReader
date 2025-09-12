import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_HEIGHT = 56
IMG_WIDTH = 256

BATCH_SIZE = 32
EPOCHS = 400

LR = 5e-4
ALPHABET = "1234567890"
idx2char = list(ALPHABET) + ["-"]
char2idx = {c: i for i, c in enumerate(idx2char)}
BLANK_IDX = len(idx2char) - 1 



