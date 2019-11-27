
#path = 'img_data/'

#vidcap = cv2.VideoCapture('match.mkv')
#success,image = vidcap.read()
#count = 0
#while success:
#  cv2.imwrite("img_data/frame%d.jpg" % count, image)     # save frame as JPEG file
#  success,image = vidcap.read()
#  print('Read a new frame: ', success)
#  count += 1

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
from PIL import Image
import os
import torch.nn as nn

class MatchFrames(Dataset):
    def __init__(self, root, transform):
        super().__init__()
        self.root = root
        self.img_names = os.listdir(self.root)
        self.transform = transform


    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = Image.open(self.root+self.img_names[idx])
        img = self.transform(img)
        noisy_img = img + torch.randn(img.shape)
        return (noisy_img, img)





train_transform = transforms.Compose([transforms.Resize((127, 127)), 
                                        transforms.ToTensor(),
                                        ])


