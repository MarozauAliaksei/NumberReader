import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class OCRDataset(Dataset):
    CHARS = '0123456789'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, data_dir, split='train', img_height=32, img_width=128):
        self.img_height = img_height
        self.img_width = img_width

        self.image_dir = os.path.join(data_dir, split, "images")
        self.file_names = sorted(os.listdir(self.image_dir))

        # Для train и val парсим текст из имени файла
        if split in ['train', 'val']:
            self.texts = [self._parse_text(fn) for fn in self.file_names]
        else:
            self.texts = None

    def _parse_text(self, file_name):
        name = os.path.splitext(file_name)[0]
        text = name[:-1]
        return text

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        file_path = os.path.join(self.image_dir, file_name)

        image = Image.open(file_path).convert('L')  # grayscale
        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image, dtype=np.float32)
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0
        image = torch.FloatTensor(image)

        if self.texts:
            text = self.texts[index]
            target = [self.CHAR2LABEL[c] for c in text if c in self.CHAR2LABEL]
            target_length = [len(target)]

            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            return image, target, target_length
        else:
            return image

def ocr_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths
