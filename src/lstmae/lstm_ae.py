import torch.nn as nn
from src.torchsnippet import ExtendNNModule

__all__ = ['LSTMAutoEncoder']

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, window_size, dropout_rate, bidirectional):
        super(EncoderLSTM, self).__init__()

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size[0],
                             num_layers=window_size, batch_first=True, bidirectional=bidirectional)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(input_size=hidden_size[0], hidden_size=hidden_size[1],
                             num_layers=window_size, batch_first=True, bidirectional=bidirectional)
        self.dropout2 = nn.Dropout(dropout_rate)
        #self.linear1 = nn.Linear(hidden_size[1], hidden_size[2])
        #self.linear2 = nn.Linear(hidden_size[2], hidden_size[3])


    def forward(self, input):
        #print('input shape  ', input.shape)
        input = input.view(input.shape[0], input.shape[1], 1)
        #print('input shape  ', input.shape)
        encoded_input, _ = self.lstm1(input)
        #encoded_input = self.dropout1(encoded_input)
        encoded_input, _ = self.lstm2(encoded_input)
        #encoded_input = self.dropout2(encoded_input)
        #print("last lstm encoder layer shape : ", encoded_input.shape)
        return encoded_input


class DecoderLSTM(nn.Module):
    def __init__(self, output_size, hidden_size, window_size, dropout_rate, bidirectional):
        super(DecoderLSTM, self).__init__()

        self.lstm1 = nn.LSTM(input_size=hidden_size[1], hidden_size=hidden_size[0],
                             num_layers=window_size, batch_first=True, bidirectional=bidirectional)
        self.dropout1 = nn.Dropout(dropout_rate)
       # self.lstm2 = nn.LSTM(input_size=hidden_size[0], hidden_size=output_size,
        #                     num_layers=window_size, batch_first=True, bidirectional=bidirectional)
        self.linear = nn.Linear(in_features=hidden_size[0], out_features=output_size)
    def forward(self, input):
        decoded_output, _ = self.lstm1(input)
        #decoded_output = self.dropout1(decoded_output)
        #decoded_output, _ = self.lstm2(decoded_output)
        decoded_output = self.linear(decoded_output)
        #print("decoded_output_linear.shape: ", decoded_output.shape)
        decoded_output = decoded_output.view(decoded_output.shape[0], decoded_output.shape[1])
        return decoded_output


class LSTMAutoEncoder(ExtendNNModule):
    def __init__(self, input_size, hidden_size, window_size, dropout_rate, bidirectional=False):
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = EncoderLSTM(input_size, hidden_size, window_size, dropout_rate, bidirectional)

        self.decoder = DecoderLSTM(input_size, hidden_size, window_size, dropout_rate, bidirectional)

    def forward(self, input):
        encoded_input = self.encoder(input)
        decoded_output = self.decoder(encoded_input)
        return decoded_output