import torch.nn as nn
import torchvision
import torch
from torchsnippet import ExtendNNModule
import torch.nn.functional as F
import numpy as np

__all__ = ['LSTMAutoEncoder']

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, window_size, bidirectional):
        super(EncoderLSTM, self).__init__()

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size[0],
                             num_layers=window_size, batch_first=True, bidirectional=bidirectional)
        self.relu1 = nn.ReLU()
        self.lstm2 = nn.LSTM(input_size=hidden_size[0], hidden_size=hidden_size[1],
                             num_layers=window_size, batch_first=True, bidirectional=bidirectional)
        self.relu2 = nn.ReLU()

    def forward(self, input):
        encoded_input, _ = self.lstm1(input)
        encoded_input = self.relu1(encoded_input)
        encoded_input, _ = self.lstm2(encoded_input)
        encoded_input = self.relu2(encoded_input)
        return encoded_input


class DecoderLSTM(nn.Module):
    def __init__(self, output_size, hidden_size, window_size, bidirectional):
        super(DecoderLSTM, self).__init__()

        self.lstm1 = nn.LSTM(input_size=hidden_size[1], hidden_size=hidden_size[0],
                             num_layers=window_size, batch_first=True, bidirectional=bidirectional)
        self.relu1 = nn.ReLU()
        self.lstm2 = nn.LSTM(input_size=hidden_size[0], hidden_size=output_size,
                             num_layers=window_size, batch_first=True, bidirectional=bidirectional)

    def forward(self, input):
        decoded_output, _ = self.lstm1(input)
        decoded_output = self.relu1(decoded_output)
        decoded_output, _ = self.lstm2(decoded_output)
        decoded_output = self.relu2()
        return decoded_output


class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, window_size, bidirectional=False):
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = EncoderLSTM(input_size, hidden_size, window_size, bidirectional)
        self.decoder = DecoderLSTM(input_size, hidden_size, window_size, bidirectional)

    def forward(self, input):
        encoded_input = self.encoder(input)
        decoded_output = self.decoder(encoded_input)
        return decoded_output