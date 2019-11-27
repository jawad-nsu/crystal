import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.input_channel = input_channel

        self.conv_block = nn.Sequential(
                                        nn.Conv2d(self.input_channel, 120, 4, stride=2, padding=1),  
                                        nn.BatchNorm2d(120),
                                        nn.ReLU(),

                                        nn.Conv2d(120, 60, 4, stride=2, padding=1),  
                                        nn.BatchNorm2d(60),
                                        nn.ReLU(),

                                        nn.Conv2d(60, 30, 4, stride=2, padding=1),
                                        nn.BatchNorm2d(30),
                                        nn.ReLU(),

                                        nn.Conv2d(30, 15, 4, stride=2, padding=1),
                                        nn.Tanh()
                                        )
        self.linear_model = nn.Sequential(nn.Linear(735, 120),
                                          nn.ReLU(),
                                          nn.Linear(120, 60),
                                          )
    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.shape[0], -1)
        x = self.linear_model(x)
        x = x.reshape(x.shape[0], 60, 1, 1)

        return x

if __name__ == '__main__':
    encoder = Encoder(3)
    img = torch.randn(16, 3, 120, 120)
    print(encoder(img).shape)





