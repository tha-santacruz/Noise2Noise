import torch
from torch import nn, optim
from torch.nn import functional as F
from pathlib import Path

class EncConvPoolBlock(nn.Module):
    """
    Encoding Convolution block : 
    2D Convolution followed by 2D Max Pooling
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
    """
    Decoding double Convoluion block :
    2 times 2D Convolution followed by Upsampling
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

class Noise2NoiseModel(nn.Module):
    """
    Noise2Noise UNet as described in the refence article
    The name of the layers in the reference architecture are commented after each corresponding module declaration
    The number of channels at each step is the same as the one in the reference article
    The final activation is linear as stated in the article. All leaky-relu have a negative slope of 0.1
    Predictions are clamped in the range [0, 255]
    """
    def __init__(self, in_channels=3, out_channels=3) -> None:
        super().__init__()
        ## Encoding layers
        self.enc_conv0 = nn.Conv2d(in_channels=in_channels, out_channels=48, kernel_size=3, padding="same") #ENC_CONV0
        self.enc_conv_pool1 = EncConvPoolBlock() #ENC_CONV1 and POOL1
        self.enc_conv_pool2 = EncConvPoolBlock() #ENC_CONV2 and POOL2
        self.enc_conv_pool3 = EncConvPoolBlock() #ENC_CONV3 and POOL3
        self.enc_conv_pool4 = EncConvPoolBlock() #ENC_CONV4 and POOL4
        self.enc_conv_pool5 = EncConvPoolBlock() #ENC_CONV5 and POOL5
        self.enc_conv6 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding="same") #ENC_CONV6
        ## Decoding layers
        self.upsample5 = nn.Upsample(scale_factor=2, mode="nearest") #UPSAMPLE5
        self.dec_conv_conv_up4 = DecConvConvUpBlock(in_channels=96) #DEC_CONV5A, DEC_CONV5B and UPSAMPLE4
        self.dec_conv_conv_up3 = DecConvConvUpBlock(in_channels=144) #DEC_CONV4A, DEC_CONV4B and UPSAMPLE3
        self.dec_conv_conv_up2 = DecConvConvUpBlock(in_channels=144) #DEC_CONV3A, DEC_CONV3B and UPSAMPLE2
        self.dec_conv_conv_up1 = DecConvConvUpBlock(in_channels=144) #DEC_CONV2A, DEC_CONV2B and UPSAMPLE1
        self.dec_conv1A = nn.Conv2d(in_channels=96+in_channels, out_channels=64, kernel_size=3, padding="same") #DEC_CONV1A
        self.dec_conv1B = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding="same") #DEC_CONV1B
        self.dec_conv1C = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, padding="same") #DEC_CONV1

    def forward(self, input):
        x0 = F.leaky_relu(self.enc_conv0(input), negative_slope=0.1)
        x1 = self.enc_conv_pool1(x0)
        x2 = self.enc_conv_pool2(x1)
        x3 = self.enc_conv_pool3(x2)
        x4 = self.enc_conv_pool4(x3)
        x5 = self.enc_conv_pool5(x4)
        x6 = F.leaky_relu(self.enc_conv6(x5), negative_slope=0.1)
        x5 = self.upsample5(x6)
        x5 = torch.cat((x5, x4), dim=1) #CONCAT5
        x4 = self.dec_conv_conv_up4(x5)
        x4 = torch.cat((x4, x3), dim=1) #CONCAT4
        x3 = self.dec_conv_conv_up3(x4)
        x3 = torch.cat((x3,x2), dim=1) #CONCAT3
        x2 = self.dec_conv_conv_up2(x3)
        x2 = torch.cat((x2, x1), dim=1) #CONCAT2
        x1 = self.dec_conv_conv_up1(x2)
        x1 = torch.cat((x1, input), dim=1) #CONCAT1
        x1 = F.leaky_relu(self.dec_conv1A(x1), negative_slope=0.1)
        x1 = F.leaky_relu(self.dec_conv1B(x1), negative_slope=0.1)
        x0 = self.dec_conv1C(x1)
        x0 = torch.clamp(x0, min=0, max=255) # added clamping to fulfill dummy input tests
        return x0

class Model():
    """
    Model declaration with related methods. Includes :
    - Pretrained model parameters loading
    - Trained model saving
    - Model training
    - Prediction with the model's current state
    - Data augmentation for training
    """
    def __init__(self) -> None:
        ## Training batch size
        self.BATCH_SIZE = 10
        ## Data augmentation for training
        self.augment_bool = True
        ## Processing device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ## Model, loss function and optimizer declaration
        self.net = Noise2NoiseModel().to(device=self.device)
        self.criterion = nn.MSELoss().to(device=self.device)
        self.optimizer = optim.Adam(self.net.parameters(), eps = 1e-8, betas=(0.9, 0.99), lr=0.00005)

    def load_pretrained_model(self, model_name = "bestmodel.pth") -> None:
        ## This loads the parameters saved in bestmodel.pth into the model
        model_path = Path(__file__).parent / model_name
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()

    def save_trained_model(self, model_name =  "newmodel.pth") -> None:
        ## This saves the parameters of the model in newmodel.pth
        model_path = Path(__file__).parent / model_name
        torch.save(self.net.state_dict(), model_path)

    def train(self, train_input, train_target, num_epochs = 100) -> None:
        ## train_input: tensor of size (N, C, H, W) containing a noisy version of the images
        ## train_target: tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs from the input by their noise
        train_input = train_input.to(device=self.device, dtype=torch.float32)
        train_target = train_target.to(device=self.device, dtype=torch.float32)
        self.net.train()
        ## Predicting and back propagating for each batch at each epoch
        for epoch in range(num_epochs):
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

    def predict(self, test_input) -> torch.Tensor:
        ## test_input : tensor of size (N1, C, H, W) that has to be denoised by the trained or loaded network
        ## Returns a tensor of the size (N1, C, H, W)
        self.net.eval()
        test_input = test_input.to(device=self.device, dtype=torch.float32)
        output = self.net(test_input)
        return output

    def augment_data(self, train_input, train_target, noise_part=0.1):
        ## This method allows to augment training data by applying simple random transformations at the batch level
        ## Gaussian noise with mean = 0 and with std = noise_part*data_std
        noise_std = torch.cat((train_input,train_target),0).std()*noise_part
        input_noise = torch.empty(train_input.size()).normal_(mean=0,std=noise_std).to(device=self.device, dtype=torch.float32)
        train_input = torch.clamp(train_input+input_noise, min=0, max=255)
        ## The two lines below can be uncommented to apply random noise on targets too.
        #target_noise = torch.empty(train_target.size()).normal_(mean=0,std=noise_std).to(device=self.device, dtype=torch.float32)
        #train_target = torch.clamp(train_target+target_noise, min=0, max=255)
        ## Random transpose
        transpose = torch.randint(0,2,()).to(torch.bool)
        if transpose:
            train_input = torch.transpose(train_input, -1, -2)
            train_target = torch.transpose(train_target, -1, -2)
        ## Random rotation
        num_rot = torch.randint(0,4,())
        train_input= torch.rot90(train_input, num_rot, [-1, -2])
        train_target= torch.rot90(train_target, num_rot, [-1, -2])
        return train_input, train_target