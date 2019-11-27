from encoder import Encoder
from decoder import Decoder
from torchvision import transforms

from torch.nn import MSELoss
from dataprep import MatchFrames, train_transform
from torch.optim import Adam
from torch.utils.data import DataLoader


from torchvision.datasets import MNIST
import torch


epoch = 25
#dataset = MatchFrames('img_dataset/')
#dataloader = DataLoader(dataset, shuffle=True, batch_size=16)
criterion = MSELoss()

mnist_dataset = MNIST('data/', train=True, transform=train_transform, download=True)
test = mnist_dataset[0][0]
mnist_dataloader = DataLoader(mnist_dataset, shuffle=True, batch_size=128)


encoder_m = Encoder(1).cuda()
decoder_m = Decoder().cuda()


encoder_optimizer = Adam(encoder_m.parameters(), lr=0.01)
decoder_optimizer = Adam(decoder_m.parameters(), lr=0.01)


def train(epochs, encoder, encoder_optimizer, decoder, decoder_optimizer, criterion, dataloader):
    batch_size = dataloader.batch_size
    steps = 0
    running_loss = 0
    for epoch in range(epochs):
        print( "*************Epoch " + str(epoch) +"*****************")

        #Training the Discriminator on real data
        for image, label in dataloader:

            """Train on real data"""
            encoder.zero_grad()
            decoder.zero_grad()
            noisy_image = image + torch.randn(image.shape)
            noisy_image = noisy_image.cuda()
            image = image.cuda()


            latent_space = encoder(noisy_image)
            recons_image = decoder(latent_space)
            loss = criterion(recons_image, image)
            running_loss += loss.item()
            steps += 1
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

        print('loss--', running_loss/steps)
        running_loss = 0
        steps = 0
        img = test.clone().detach()
        img = img + torch.randn(img.shape)
        img = img.reshape(1, 1, 127, 127)
        img = img.cuda()
        output = decoder(encoder(img).reshape(1, 60, 1, 1))
        output = output.reshape(1, 127, 127).cpu()
        transform = transforms.ToPILImage()
        img_file = transform(output)
        img_file.save('results/test'+str(epoch)+'.jpg')

        
            

                   
train(42, encoder_m, encoder_optimizer, decoder_m, decoder_optimizer, criterion, mnist_dataloader)
 

