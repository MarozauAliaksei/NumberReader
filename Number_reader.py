import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
# -----------------
# 1. Трансформации
# -----------------
class DigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DigitCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # -----------------
    # 2. Датасет и деление
    # -----------------
    full_dataset = datasets.ImageFolder(root="data/Numbers", transform=transform)

    val_ratio = 0.2
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print("Классы:", full_dataset.classes)  # ['0','1',...,'9']

    # -----------------
    # 3. CNN для классификации цифр
    # -----------------


    # -----------------
    # 4. Обучение
    # -----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    tot_los = 1231
    epoch = 0
    while  tot_los > 1e-2:
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            tot_los = total_loss/len(train_loader)
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}")
        epoch += 1

    torch.save(model.state_dict(), "digit_pretrain.pth")

    img_path = "data/testNum/0.png"
    image = Image.open(img_path)
    image = transform(image).unsqueeze(0).to(device) 

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()

    print(f"Файл: {img_path}, предсказание: {pred}")