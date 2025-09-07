import torch
import torch.nn as nn
from config import IMG_HEIGHT

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1),(2,1)),

            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1),(2,1)),

            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1),(2,1)),

            nn.Conv2d(256,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((1,None))
        )

        self.rnn = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self,x):
        conv = self.cnn(x)
        b,c,h,w = conv.size()
        assert h==1
        conv = conv.squeeze(2)           # [B,C,W]
        conv = conv.permute(2,0,1)       # [W,B,C]
        rnn_out,_ = self.rnn(conv)       # [W,B,512]
        out = self.fc(rnn_out)           # [W,B,num_classes]
        return out
