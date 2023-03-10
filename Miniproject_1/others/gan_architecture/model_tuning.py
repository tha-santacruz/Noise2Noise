import torch
from torch import nn, optim
from torch.nn import functional as F
import time
import os
from pathlib import Path

class EncConvPoolBlock(nn.Module):
    """Noise2Noise Unet encoder block
    """
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
    """Noise2Noise Unet decoder block
    """
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


class GNet(nn.Module):
    """Noise2Noise Unet model, 
    Generator of the GAN
    """
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        ## Encoding layers
        self.enc_conv0 = nn.Conv2d(in_channels=in_channels, out_channels=48, kernel_size=3, padding="same")
        self.enc_conv_pool1 = EncConvPoolBlock()
        self.enc_conv_pool2 = EncConvPoolBlock()
        self.enc_conv_pool3 = EncConvPoolBlock()
        self.enc_conv_pool4 = EncConvPoolBlock()
        self.enc_conv_pool5 = EncConvPoolBlock()
        self.enc_conv6 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding="same")

        ## Decoding layers
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


class CBlock(nn.Module):
    """PatchGAN Conv-BatchNorm-LeakyReLU block
    """
    def __init__(self, in_channels, out_channels, use_batchnorm):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            )
        if use_batchnorm:
            self.block.append(nn.BatchNorm2d(out_channels))

    def forward(self, input):
        return self.block(input)


class DNet(nn.Module):
    """PatchGAN network,
    Discriminator of the GAN,
    Architecture : C64-C128-C256
    """
    def __init__(self, in_channels=6):
        super().__init__()
        self.C64 = CBlock(in_channels=6, out_channels=64, use_batchnorm=False)
        self.C128 = CBlock(in_channels=64, out_channels=128, use_batchnorm=True)
        self.C256 = CBlock(in_channels=128, out_channels=256, use_batchnorm=True)
        self.C1 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False)

    def forward(self, input):
        output = F.leaky_relu(self.C64(input))
        output = F.leaky_relu(self.C128(output))
        output = F.leaky_relu(self.C256(output))
        output = torch.sigmoid(self.C1(output))
        return output


class Model():
    def __init__(self, lr = 1e-3, betas = (0.9, 0.99), eps = 1e-8, lossf = "L2", augment = True, batch_size = 100, lf = 0.5, bluring = 0) -> None:
        ## Setting batch size (same as the testing batch size from the train.py file)
        self.BATCH_SIZE = batch_size
        ## Setting loss factor to weight the generator loss
        self.LOSS_FACTOR = lf
        ## Selecting device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        ## Generator network
        self.G_net = GNet().to(device=self.device)
        if lossf == "L1":
            self.G_criterion = nn.L1Loss().to(device=self.device)
        elif lossf == "L2":
            self.G_criterion = nn.MSELoss().to(device=self.device)
        else:
            raise ValueError("invalid generator loss function. valid options are L1 and L2")

        self.G_optimizer = optim.Adam(self.G_net.parameters(), eps = eps, betas=betas, lr=lr)

        ## Discriminator network
        self.D_net = DNet().to(device=self.device)
        self.D_criterion = nn.BCELoss().to(device=self.device) 
        self.D_optimizer = optim.Adam(self.D_net.parameters(), eps = eps, betas=betas, lr=lr)

        ## For logs
        self.total_train_time = 0
        self.total_train_epochs = 0

        ## Data augmentation
        self.augment_bool = augment
        self.kernel_center = 1-bluring
        self.kernel_other = bluring/8
        self.blur_kernel = torch.tensor([[self.kernel_other,self.kernel_other,self.kernel_other],
                                        [self.kernel_other,self.kernel_center,self.kernel_other],
                                        [self.kernel_other,self.kernel_other,self.kernel_other]]).to(device=self.device).to(torch.float32)
        self.filter_kernel = torch.zeros(3,3,3,3).to(device=self.device)
        for i in range(3):
            self.filter_kernel[i,i,:,:] = self.blur_kernel

    def load_pretrained_model(self, model_name = "bestmodel.pth") -> None:
        ## This loads the parameters saved in bestmodel.pth into the model
        model_path = Path(__file__).parent / model_name
        self.G_net.load_state_dict(torch.load(model_path))
        self.G_net.eval()

    def save_trained_model(self, model_name =  "newmodel.pth") -> None:
        ## This saves the parameters of the model
        model_path = Path(__file__).parent / model_name
        torch.save(self.G_net.state_dict(), model_path)

    def train(self, inputs_1, inputs_2, num_epochs = 100) -> None:
        ## input_1: tensor of size (N, C, H, W) containing a noisy version of the images
        ## input_2: tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs from the input_1 by their noise
        logs = []
        inputs_1 = inputs_1.to(device=self.device, dtype=torch.float32)
        inputs_2 = inputs_2.to(device=self.device, dtype=torch.float32)
        self.G_net.train()
        self.D_net.train()
        true_labels = torch.ones([self.BATCH_SIZE,1,1,1]).to(device=self.device, dtype=torch.float32)
        false_labels = torch.zeros([self.BATCH_SIZE,1,1,1]).to(device=self.device, dtype=torch.float32)

        ## Train loop
        for epoch in range(num_epochs):
            accumulated_G_loss = 0
            accumulated_D_loss = 0
            time_before = time.time()

            ## Infer batch and backprop
            for b in range(0, inputs_1.size(0), self.BATCH_SIZE):
                ## Zero grat optims
                self.D_optimizer.zero_grad()
                self.G_optimizer.zero_grad()
                ## Get data
                input_1 = inputs_1.narrow(0, b, self.BATCH_SIZE)
                input_2 = inputs_2.narrow(0, b, self.BATCH_SIZE)
                if self.augment_bool:
                    input_1, input_2 = self.augment_data(input_1=input_1, input_2=input_2)

                ## Generate fake image
                input_2_fake = self.G_net(input_1)

                ## Blur real image to make it harder for the D_net
                input_2_blurred = self.blur_target(input_2)

                ## Get loss of the D_net and backprop
                real_pair = torch.cat((input_1, input_2_blurred),dim=1)
                fake_pair = torch.cat((input_1, input_2_fake.detach()),dim=1)
                D_loss = (self.D_criterion(self.D_net(real_pair),true_labels) + self.D_criterion(self.D_net(fake_pair),false_labels))*0.5
                D_loss.backward()
                self.D_optimizer.step()

                ## No grad to avoid backprop on D
                for param in self.D_net.parameters():
                    param.requires_grad = False

                ## Get the loss ot the G_net
                fake_pair = torch.cat((input_1, input_2_fake),dim=1)
                G_loss = self.D_criterion(self.D_net(fake_pair),true_labels)*self.LOSS_FACTOR + self.G_criterion(input_2_fake, input_2)*(1-self.LOSS_FACTOR)
                G_loss.backward()
                self.G_optimizer.step()

                ## Restore backprop to D
                for param in self.D_net.parameters():
                    param.requires_grad = True

                accumulated_G_loss += G_loss.item()
                accumulated_D_loss += D_loss.item()

            ## create logs
            epoch_time = time.time()-time_before
            self.total_train_time += epoch_time
            self.total_train_epochs += 1
            logs.append(torch.tensor([self.total_train_epochs, accumulated_G_loss, accumulated_D_loss, self.total_train_time]))

            ## create chekpoint
            G_model_path = Path(__file__).parent / "checkpoints" / f"checkpoint_G_{self.total_train_epochs}.pth"
            torch.save(self.G_net.state_dict(), G_model_path)
            D_model_path = Path(__file__).parent / "checkpoints" / f"checkpoint_D_{self.total_train_epochs}.pth"
            torch.save(self.G_net.state_dict(), D_model_path)
        logs = torch.stack(logs)
        return logs

    def predict(self, test_input) -> torch.Tensor:
        ## test_input : tensor of size (N1, C, H, W) that has to be denoised by the trained or loaded network
        ## Returns a tensor of the size (N1, C, H, W)
        self.G_net.eval()
        test_input = test_input.to(device=self.device, dtype=torch.float32)
        output = self.G_net(test_input)
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

    def augment_data(self, input_1, input_2, noise_part=0.1):
        # This method allows to augment training data by applying simple random transformations at the batch level
        # Gaussian noise with mean = 0 and with std = noise_part*data_std
        noise_std = torch.cat((input_1,input_2),0).std()*noise_part
        input_noise = torch.empty(input_1.size()).normal_(mean=0,std=noise_std).to(device=self.device, dtype=torch.float32)
        target_noise = torch.empty(input_2.size()).normal_(mean=0,std=noise_std).to(device=self.device, dtype=torch.float32)
        input_1 = torch.clamp(input_1+input_noise, min=0, max=255)
        input_2 = torch.clamp(input_2+target_noise, min=0, max=255)
        # Random transpose
        transpose = torch.randint(0,2,()).to(torch.bool)
        if transpose:
            input_1 = torch.transpose(input_1, -1, -2)
            input_2 = torch.transpose(input_2, -1, -2)
        # Random rotation
        num_rot = torch.randint(0,4,())
        input_1= torch.rot90(input_1, num_rot, [-1, -2])
        input_2= torch.rot90(input_2, num_rot, [-1, -2])
        return input_1, input_2

    def blur_target(self,target):
        blurred_target = F.conv2d(target,self.filter_kernel,padding=1)
        return blurred_target

def psnr(denoised , ground_truth):
    ## Peak Signal to Noise Ratio: denoised and ground ̇truth have range [0, 1] 
    mse = torch.mean((denoised - ground_truth) ** 2)
    return -10 * torch.log10(mse + 10**-8)