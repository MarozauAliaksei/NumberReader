import torch
from dataset import OCRDataset, collate_fn
from torch.utils.data import DataLoader
from main import model  # твоя модель
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import string

# ------------------------------
# Настройка
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Символы и декодер CTC
alphabet = string.digits  # '0123456789'
blank_char = '-'  # если у тебя blank токен отдельный
idx2char = {i: c for i, c in enumerate(alphabet)}

def ctc_decode(pred):
    # pred: T x C (логиты)
    pred = pred.argmax(1).detach().cpu().numpy()
    # удаляем повторения и blank
    decoded = []
    previous = None
    for p in pred:
        if p != previous and p < len(alphabet):
            decoded.append(idx2char[p])
        previous = p
    return ''.join(decoded)

# ------------------------------
# Загрузка картинки
# ------------------------------
img_path = "data/train/images/263_.jpg"  # твоя картинка
label = "263"  # настоящая метка

# Преобразование как при обучении
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

img = Image.open(img_path).convert("L")
img = transform(img).unsqueeze(0).to(device)  # 1 x 1 x H x W

# ------------------------------
# Прогон через модель
# ------------------------------
with torch.no_grad():
    output = model(img)  # обычно: B x C x T
    output = output.squeeze(0).permute(1, 0)  # T x C, как надо для CTC
    pred_label = ctc_decode(output)

# ------------------------------
# Вывод
# ------------------------------
print(f"Истинная метка: {label}")
print(f"Предсказание модели: {pred_label}")

# Визуализация
plt.imshow(img.squeeze().cpu(), cmap='gray')
plt.title(f"GT: {label} | Pred: {pred_label}")
plt.axis('off')
plt.show()
