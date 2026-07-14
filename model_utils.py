# -*- coding: utf-8 -*-
# Author:Yu Chen
# Data: 2025/9/22 10:17
# Email: yu2000.chen@connect.polyu.hk


import torch
import torch.nn as nn
import torch.nn.functional as F

class Normal_encoder(nn.Module):
    def __init__(self, resize, in_channels, out_channels):
        super(Normal_encoder, self).__init__()
        # input 32 1 128 * 128 -> 32 14
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resize = resize
        ##
        nn_size = self.calculate_size(self.resize, 4, 2, 1)
        nn_size = self.calculate_size(nn_size, 4, 2, 1)
        nn_size = self.calculate_size(nn_size, 4, 2, 1)
        nn_size = self.calculate_size(nn_size, 4, 2, 1)
        # nn_size must be integer
        assert (nn_size % 1 == 0)
        nn_size = int(nn_size)
        self.encoder_cnn_1 = nn.Sequential(
            self.cnn_block(in_channels, 2 * out_channels, 4, 2, 1),
            self.cnn_block(2 * out_channels, 4 * out_channels, 4, 2, 1),
            self.cnn_block(4 * out_channels, 2 * out_channels, 4, 2, 1),
            self.cnn_block(2 * out_channels, 1, 4, 2, 1),
            nn.Flatten(),
            self.nn_block(in_features=nn_size * nn_size, out_features = 128),
            nn.Dropout(),
            self.nn_block(in_features=128, out_features=7),
        )
        self.encoder_cnn_2 = nn.Sequential(
            self.cnn_block(in_channels, 2 * out_channels, 4, 2, 1),
            self.cnn_block(2 * out_channels, 4 * out_channels, 4, 2, 1),
            self.cnn_block(4 * out_channels, 2 * out_channels, 4, 2, 1),
            self.cnn_block(2 * out_channels, 1, 4, 2, 1),
            nn.Flatten(),
            self.nn_block(in_features=nn_size * nn_size, out_features=128),
            nn.Dropout(),
            self.nn_block(in_features=128, out_features=7),
        )
        self.final_layer = nn.Linear(in_features=14, out_features=7)

    def calculate_size(self, pixel_size, kernel_size, stride, padding):
        size = (pixel_size + 2 * padding - 1 * (kernel_size - 1) - 1) / stride + 1
        return size

    def cnn_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )
    def nn_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(0.2),
        )

    def forward(self, Gain, phase):
        gain = self.encoder_cnn_1(Gain)
        phase = self.encoder_cnn_2(phase)
        y = torch.concat([gain, phase], dim=1)
        y = self.final_layer(y)
        y = nn.functional.relu(y)
        return y


class Normal_encoder_1(nn.Module):
    def __init__(self, resize, in_channels, out_channels):
        super(Normal_encoder_1, self).__init__()
        # input 32 1 128 * 128 -> 32 14
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resize = resize
        ##
        nn_size = self.calculate_size(self.resize, 4, 2, 1)
        nn_size = self.calculate_size(nn_size, 4, 2, 1)
        nn_size = self.calculate_size(nn_size, 4, 2, 1)
        nn_size = self.calculate_size(nn_size, 4, 2, 1)
        # nn_size must be integer
        assert (nn_size % 1 == 0)
        nn_size = int(nn_size)
        self.encoder_cnn_1 = nn.Sequential(
            self.cnn_block(in_channels, 2 * out_channels, 4, 2, 1),
            self.cnn_block(2 * out_channels, 4 * out_channels, 4, 2, 1),
            self.cnn_block(4 * out_channels, 2 * out_channels, 4, 2, 1),
            self.cnn_block(2 * out_channels, 1, 4, 2, 1),
            nn.Flatten(),
            self.nn_block(in_features=nn_size * nn_size, out_features = 128),
            nn.Dropout(),
            self.nn_block(in_features=128, out_features=7),
        )
        self.encoder_cnn_2 = nn.Sequential(
            self.cnn_block(in_channels, 2 * out_channels, 4, 2, 1),
            self.cnn_block(2 * out_channels, 4 * out_channels, 4, 2, 1),
            self.cnn_block(4 * out_channels, 2 * out_channels, 4, 2, 1),
            self.cnn_block(2 * out_channels, 1, 4, 2, 1),
            nn.Flatten(),
            self.nn_block(in_features=nn_size * nn_size, out_features=128),
            nn.Dropout(),
            self.nn_block(in_features=128, out_features=7),
        )
        self.final_layer = nn.Linear(in_features=14, out_features=7)

    def calculate_size(self, pixel_size, kernel_size, stride, padding):
        size = (pixel_size + 2 * padding - 1 * (kernel_size - 1) - 1) / stride + 1
        return size

    def cnn_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )
    def nn_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(0.2),
        )

    def forward(self, Gain, phase):
        gain = self.encoder_cnn_1(Gain)
        phase = self.encoder_cnn_2(phase)
        y = torch.concat([gain, phase], dim=1)
        y = self.final_layer(y)
        return y


class Auxiliary_model(nn.Module):
    def __init__(self, loss_sequence_size, output_size=2):
        super(Auxiliary_model, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=loss_sequence_size, batch_first=True)
        self.fc = nn.Linear(loss_sequence_size, output_size)

    def forward(self, x):
        # x: (batch_size, loss_sequence_size, 1)
        x = torch.unsqueeze(x, 2)
        out, (h_n, c_n) = self.lstm(x)
        # Take the last hidden state: (batch_size, 15)
        out = out[:, -1, :]
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        return out


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def text_autoencoder():
    encoder = Normal_encoder(128, 1, 2)
    initialize_weights(encoder)
    a = torch.rand(size=(1, 1, 128, 128))
    y = encoder(a, a)
    print(y)

if __name__ == "__main__":
    text_autoencoder()