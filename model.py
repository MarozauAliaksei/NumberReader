import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, in_channel=1, num_classes=11, cnn_input_height=32, rnn_hidden=64, use_gru=True, dropout=0.1):
        super().__init__()
        self.cnn_input_height = cnn_input_height

        # ---------- CNN (очень компактная)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channel, 16, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # downsample
            nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True)   # downsample
        )

        cnn_output_height = cnn_input_height // 2 // 2
        rnn_input_size = 64 * cnn_output_height

        # ---------- RNN
        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        ) if use_gru else nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.fc = nn.Linear(rnn_hidden*2, num_classes)

    def forward(self, x):
        x = self.cnn(x)  # [B, C, H, W]
        x = F.adaptive_avg_pool2d(x, (self.cnn_input_height//4, None))
        B, C, H, W = x.shape
        x = x.permute(0,3,1,2).contiguous()  # [B, W, C, H]
        x = x.view(B, W, C*H)

        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

# ----- Тест -----
if __name__ == "__main__":
    data = torch.randn(4, 1, 32, 160)
    model = TinyCRNN()
    out = model(data)
    print("Output shape:", out.shape)
    print(f"Approx model size: {sum(p.numel() for p in model.parameters())*4/1024/1024:.2f} MB")
