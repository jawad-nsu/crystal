import torch.nn as nn
import torch


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block = nn.Sequential(
                                        nn.ConvTranspose2d(60, 120, 5, stride=2, padding=1),  
                                        nn.BatchNorm2d(120),
                                        nn.ReLU(),

                                        nn.ConvTranspose2d(120, 60, 5, stride=2, padding=1),  
                                        nn.BatchNorm2d(60),
                                        nn.ReLU(),

                                        nn.ConvTranspose2d(60, 30, 5, stride=2, padding=1),
                                        nn.BatchNorm2d(30),
                                        nn.ReLU(),

                                        nn.ConvTranspose2d(30, 3, 5, stride=2, padding=1),
                                        nn.Tanh(),
                                        nn.ConvTranspose2d(3, 3, 5, stride=2, padding=1),
                                        nn.ConvTranspose2d(3, 1, 5, stride=2, padding=1),
                                        )


    def forward(self, x):
        x = self.conv_block(x)

        return x

if __name__ == '__main__':
    encoder = Decoder()
    img = torch.randn(1, 60, 1, 1)
    
    print(encoder(img).shape)


