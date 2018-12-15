import torch.nn as nn
from src.torchsnippet import ExtendNNModule

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

        self.linear1 = nn.Linear(hidden_size[1], hidden_size[2])
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size[2], hidden_size[3])
        self.relu4 = nn.ReLU()


    def forward(self, input):
        #print('input shape  ', input.shape)
        input = input.view(input.shape[0], input.shape[1], 1)
        #print('input shape  ', input.shape)
        encoded_input, _ = self.lstm1(input)
        encoded_input = self.relu1(encoded_input)
        encoded_input, _ = self.lstm2(encoded_input)
        encoded_input = self.relu2(encoded_input)
        print("last lstm encoder layer shape : ", encoded_input.shape)

        #encoded_linear = encoded_input.view(encoded_input.size(0), -1)
        #encoded_linear = self.linear1(encoded_linear)
        #encoded_linear = self.relu3(encoded_linear)
        #encoded_linear = self.linear2(encoded_linear)
        #encoded_linear = self.relu34(encoded_linear)
        return encoded_input


class DecoderLSTM(nn.Module):
    def __init__(self, output_size, hidden_size, window_size, bidirectional):
        super(DecoderLSTM, self).__init__()

        self.lstm1 = nn.LSTM(input_size=hidden_size[1], hidden_size=hidden_size[0],
                             num_layers=window_size, batch_first=True, bidirectional=bidirectional)
        self.relu1 = nn.ReLU()
        self.lstm2 = nn.LSTM(input_size=hidden_size[0], hidden_size=output_size,
                             num_layers=window_size, batch_first=True, bidirectional=bidirectional)
        self.relu2 = nn.ReLU()

    def forward(self, input):
        decoded_output, _ = self.lstm1(input)
        decoded_output = self.relu1(decoded_output)
        decoded_output, _ = self.lstm2(decoded_output)
        decoded_output = self.relu2(decoded_output)
        decoded_output = decoded_output.view(decoded_output.shape[0], decoded_output.shape[1])
        return decoded_output


class LSTMAutoEncoder(ExtendNNModule):
    def __init__(self, input_size, hidden_size, window_size, bidirectional=False):
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = EncoderLSTM(input_size, hidden_size, window_size, bidirectional)

        self.decoder = DecoderLSTM(input_size, hidden_size, window_size, bidirectional)

    def forward(self, input):
        encoded_input = self.encoder(input)
        decoded_output = self.decoder(encoded_input)
        return decoded_output