import torch
from torch import nn, optim
from torch.nn import functional as F
import time
import os
from pathlib import Path

## Imports to delete : time, matplotlib.pyplot

class EncConvPoolBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding="same")
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        y = self.conv(x)
        y = F.leaky_relu(y, negative_slope=0.1)
        y = self.maxpool(y)
        return y

class DecConvConvUpBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.convA = nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=3, padding="same")
        self.convB = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding="same")
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
    
    def forward(self, x):
        y = self.convA(x)
        y = F.leaky_relu(y, negative_slope=0.1)
        y = self.convB(y)
        y = F.leaky_relu(y, negative_slope=0.1)
        y = self.upsample(y)
        return y

class Noise2NoiseModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3) -> None:
        super().__init__()
        # Encoding layers
        self.enc_conv0 = nn.Conv2d(in_channels=in_channels, out_channels=48, kernel_size=3, padding="same")
        self.enc_conv_pool1 = EncConvPoolBlock()
        self.enc_conv_pool2 = EncConvPoolBlock()
        self.enc_conv_pool3 = EncConvPoolBlock()
        self.enc_conv_pool4 = EncConvPoolBlock()
        self.enc_conv_pool5 = EncConvPoolBlock()
        self.enc_conv6 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding="same")
        # Decoding layers
        self.upsample5 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec_conv_conv_up4 = DecConvConvUpBlock(in_channels=96)
        self.dec_conv_conv_up3 = DecConvConvUpBlock(in_channels=144)
        self.dec_conv_conv_up2 = DecConvConvUpBlock(in_channels=144)
        self.dec_conv_conv_up1 = DecConvConvUpBlock(in_channels=144)
        self.dec_conv1A = nn.Conv2d(in_channels=96+in_channels, out_channels=64, kernel_size=3, padding="same")
        self.dec_conv1B = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding="same")
        self.dec_conv1C = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, padding="same")

    def forward(self, input):
        x0 = F.leaky_relu(self.enc_conv0(input), negative_slope=0.1)
        x1 = self.enc_conv_pool1(x0)
        x2 = self.enc_conv_pool2(x1)
        x3 = self.enc_conv_pool3(x2)
        x4 = self.enc_conv_pool4(x3)
        x5 = self.enc_conv_pool5(x4)
        x6 = F.leaky_relu(self.enc_conv6(x5), negative_slope=0.1)
        x5 = self.upsample5(x6)
        x5 = torch.cat((x5, x4), dim=1)
        x4 = self.dec_conv_conv_up4(x5)
        x4 = torch.cat((x4, x3), dim=1)
        x3 = self.dec_conv_conv_up3(x4)
        x3 = torch.cat((x3,x2), dim=1)
        x2 = self.dec_conv_conv_up2(x3)
        x2 = torch.cat((x2, x1), dim=1)
        x1 = self.dec_conv_conv_up1(x2)
        x1 = torch.cat((x1, input), dim=1)
        x1 = F.leaky_relu(self.dec_conv1A(x1), negative_slope=0.1)
        x1 = F.leaky_relu(self.dec_conv1B(x1), negative_slope=0.1)
        x0 = self.dec_conv1C(x1)
        x0 = torch.clamp(x0, min=0, max=255) # added clamping to fulfill dummy input tests
        return x0


class Model():
    def __init__(self, lr, betas, eps, lossf, augment = True) -> None:
        self.BATCH_SIZE = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.net = Noise2NoiseModel().to(device=self.device)
        self.criterion = lossf().to(device=self.device)
        self.optimizer = optim.Adam(self.net.parameters(), eps = eps, betas=betas, lr=lr)

        self.total_train_time = 0
        self.total_train_epochs = 0

        self.augment_bool = augment

    def load_pretrained_model(self, model_name = "bestmodel.pth") -> None:
        ## This loads the parameters saved in bestmodel.pth into the model
        model_path = Path(__file__).parent / model_name
        #model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),model_name)
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()

    def save_trained_model(self, model_name =  "newmodel.pth") -> None:
        ## This saves the parameters of the model
        model_path = Path(__file__).parent / model_name
        torch.save(self.net.state_dict(), model_path)

    def train(self, train_input, train_target, num_epochs = 100) -> None:
        ## train_input: tensor of size (N, C, H, W) containing a noisy version of the images
        ## train_target: tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs from the input by their noise
        logs = []
        train_input = train_input.to(device=self.device, dtype=torch.float32)
        train_target = train_target.to(device=self.device, dtype=torch.float32)
        for epoch in range(num_epochs):
            self.net.train()
            accumulated_loss = 0
            time_before = time.time()
            for b in range(0, train_input.size(0), self.BATCH_SIZE):
                input = train_input.narrow(0, b, self.BATCH_SIZE)
                target = train_target.narrow(0, b, self.BATCH_SIZE)
                if self.augment_bool:
                    input, target = self.augment_data(train_input=input, train_target=target)
                output = self.net(input)
                loss = self.criterion(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                accumulated_loss += loss.item()
            epoch_time = time.time()-time_before
            self.total_train_time += epoch_time
            self.total_train_epochs += 1
            logs.append(torch.tensor([self.total_train_epochs,accumulated_loss,self.total_train_time]))
        logs = torch.stack(logs)
        return logs

    def predict(self, test_input) -> torch.Tensor:
        ## test_input : tensor of size (N1, C, H, W) that has to be denoised by the trained or loaded network
        ## Returns a tensor of the size (N1, C, H, W)
        self.net.eval()
        test_input = test_input.to(device=self.device, dtype=torch.float32)
        output = self.net(test_input)
        return output

    def evaluate(self, test_input, test_target):
        psnrs = torch.Tensor().to(device=self.device)
        test_target = test_target.to(device=self.device, dtype=torch.float32)
        for b in range(0, test_input.size(0), self.BATCH_SIZE):
            input = test_input.narrow(0, b, self.BATCH_SIZE)
            target = test_target.narrow(0, b, self.BATCH_SIZE)
            output = self.predict(input)
            psnrs = torch.cat((psnrs, psnr(output, target).view(-1)))
        print("psnr : {}".format(psnrs.mean()))
        return psnrs.mean()

    def augment_data(self, train_input, train_target, noise_part=0.1):
        # This method allows to augment training data by applying simple random transformations at the batch level
        # Gaussian noise with mean = 0 and with std = noise_part*data_std
        noise_std = torch.cat((train_input,train_target),0).std()*noise_part
        input_noise = torch.empty(train_input.size()).normal_(mean=0,std=noise_std).to(device=self.device, dtype=torch.float32)
        #target_noise = torch.empty(train_target.size()).normal_(mean=0,std=noise_std).to(device=self.device, dtype=torch.float32)
        train_input = torch.clamp(train_input+input_noise, min=0, max=255)
        #train_target = torch.clamp(train_target+target_noise, min=0, max=255)
        # Random transpose
        transpose = torch.randint(0,2,()).to(torch.bool)
        if transpose:
            train_input = torch.transpose(train_input, -1, -2)
            train_target = torch.transpose(train_target, -1, -2)
        # Random rotation
        num_rot = torch.randint(0,4,())
        train_input= torch.rot90(train_input, num_rot, [-1, -2])
        train_target= torch.rot90(train_target, num_rot, [-1, -2])
        return train_input, train_target



def psnr(denoised , ground_truth):
    ## Peak Signal to Noise Ratio: denoised and ground ̇truth have range [0, 1] 
    mse = torch.mean((denoised - ground_truth) ** 2)
    return -10 * torch.log10(mse + 10**-8)