# model_crnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *


class CRNN(nn.Module):
    def __init__(self, num_classes, dropout_cnn=0.2, dropout_rnn=0.3):
        super(CRNN, self).__init__()
        # ----------------- CNN Backbone -----------------
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.norm3 = nn.InstanceNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.norm4 = nn.InstanceNorm2d(64)
        self.drop_cnn = nn.Dropout2d(dropout_cnn)

        # ----------------- RNN -----------------
        self.gru_input_size = (IMG_HEIGHT // 4) * 64
        self.gru_hidden_size = 128
        self.gru_layers = 2
        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=self.gru_hidden_size,
            num_layers=self.gru_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rnn
        )

        # ----------------- FC -----------------
        self.fc = nn.Linear(self.gru_hidden_size * 2, num_classes)

    def forward(self, x):
        # x: [B,1,H,W]
        x = F.leaky_relu(self.norm1(self.conv1(x)))# Заменить на бэтч нормы скип конеккшон вместо дропаут
        x = F.leaky_relu(self.norm2(self.conv2(x)))
        x = self.drop_cnn(x)
        x = F.leaky_relu(self.norm3(self.conv3(x)))
        x = F.leaky_relu(self.norm4(self.conv4(x)))
        x = self.drop_cnn(x)

        b, c, h, w = x.size()  # [B,C,H,W]

        # [B,C,H,W] -> [B,W,H,C] -> [B,W,H*C] для GRU не нужон
        x = x.permute(0, 3, 2, 1).reshape(b, w, h * c)

        # GRU
        x, _ = self.gru(x)  # [B,W,2*hidden]
        x = self.fc(x)
        x = F.log_softmax(x, dim=-1)
        x = x.permute(1,0,2)
        return x


# --------- Тестовый запуск ---------
if __name__ == "__main__":
    model = CRNN(num_classes=11)
    data = torch.randn(64,1,28,140)
    out = model(data)
    print(f"Input: {data.shape}, Output: {out.shape}")  # [W,B,num_classes]
