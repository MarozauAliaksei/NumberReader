import torch
from PIL import Image
import torchvision.transforms as transforms
from model import CRNN
from Dataset import OCRDataset  # чтобы использовать LABEL2CHAR
from config import config as config

# -----------------------------
# Конфигурация
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_height = config['img_height']
img_width = config['img_width']

# Путь к сохраненной модели
checkpoint_path = "./checkpoint/crnn.pt"  # укажи свой путь

# -----------------------------
# Создаем модель и загружаем веса
# -----------------------------
num_class = len(OCRDataset.LABEL2CHAR) + 1
crnn = CRNN(
    1, img_height, img_width, num_class,
    map_to_seq_hidden=128,
    rnn_hidden=256,
    leaky_relu=True
)
crnn.load_state_dict(torch.load(checkpoint_path, map_location=device))
crnn.to(device)
crnn.eval()

# -----------------------------
# Функция для предсказания числа с изображения
# -----------------------------
def predict_image(image_path):
    # Загружаем и преобразуем изображение
    image = Image.open(image_path).convert("L")  # конвертируем в grayscale
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image).unsqueeze(0).to(device)  # добавляем batch dimension

    # Прямой проход через модель
    with torch.no_grad():
        logits = crnn(image)
        preds = torch.argmax(logits, dim=2).permute(1, 0)  # (batch, seq_len)

    # Greedy-декодирование
    result = []
    for seq in preds:
        string = []
        prev = -1
        for p in seq:
            p = p.item()
            if p != prev and p != 0:  # 0 = blank
                string.append(OCRDataset.LABEL2CHAR[p])
            prev = p
        result.append("".join(string))

    return result[0]

# -----------------------------
# Тестируем на изображении
# -----------------------------
image_path = "./data/test/images/42_.jpg"  # укажи путь к картинке
number = predict_image(image_path)
print("Распознанное число:", number)
