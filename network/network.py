import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        
        # Encoder部分: 3层CNN
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 输出尺寸: (64, 128, 128)
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 输出尺寸: (128, 64, 64)
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 输出尺寸: (256, 32, 32)
            nn.ReLU(True)
        )
        
        # Decoder部分: 3层Deconvolutional layer (转置卷积)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # 输出尺寸: (128, 64, 64)
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输出尺寸: (64, 128, 128)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),    # 输出尺寸: (3, 256, 256)
            nn.Sigmoid()  # 如果你的图像数据被归一化到[0,1]，使用Sigmoid来确保输出也在这个范围
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
