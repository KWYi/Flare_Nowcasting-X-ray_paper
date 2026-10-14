import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

class Encoder4Attn(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num, encoded_size=False, drop_frac=0.0,
                 initiallize=False, bidirection=False):
        super(Encoder4Attn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoded_size = encoded_size
        self.layer_num = layer_num
        self.bidirection = bidirection

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.layer_num,
                            batch_first=True, dropout=drop_frac, bidirectional=self.bidirection)

        lstm_output_size = self.hidden_size * (2 if self.bidirection else 1)

        if self.encoded_size:
            self.linear = [nn.Linear(in_features=lstm_output_size, out_features=self.encoded_size),
                           nn.ReLU()]
            self.linear = nn.Sequential(*self.linear)

        if initiallize:
            self.lstm.apply(self.init_weights)
            if self.encoded_size:
                self.linear_act_name = 'relu'
                self.linear.apply(self.init_weights)

    def forward(self, input):

        x, hidden = self.lstm(input)
        if self.encoded_size:
            x = self.linear(x)

        return x, hidden

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.linear_act_name == 'relu' or self.linear_act_name == 'leaky_reLU':
                nn.init.kaiming_normal_(module.weight.detach(), mode='fan_out', nonlinearity=self.linear_act_name)
                # nn.init.kaiming_uniform_(module.weight.detach(), mode='fan_out', nonlinearity='relu')
                module.bias.detach().fill_(0.)
            elif any(self.linear_act_name == np.array(['tanh', 'sigmoid', 'linear'])):
                nn.init.xavier_normal_(module.weight.detach(), gain=nn.init.calculate_gain(self.linear_act_name))
                # nn.init.xavier_uniform_(module.weight.detach(), gain=nn.init.calculate_gain(self.linear_act_name))
                module.bias.detach().fill_(0.)
            else:
                raise NotImplementedError("Initializer got error!")
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)


class AttnDecoder_ver3(nn.Module):
    def __init__(self, encoded_size, hidden_size, layer_num, input_len, output_size=1, drop_frac=0.,
                 extracted_size=False, initiallize=False):
        super(AttnDecoder_ver3, self).__init__()
        self.encoded_size = encoded_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.input_len = input_len
        self.extracted_size = extracted_size
        self.output_size = output_size

        self.fc_hidden = nn.Linear(self.hidden_size, self.encoded_size, bias=False)
        self.fc_encoder = nn.Linear(self.encoded_size, self.encoded_size, bias=False)
        self.attention_weights = nn.Linear(self.encoded_size, 1)
        self.lstm = nn.LSTM(input_size=self.encoded_size + 1, hidden_size=self.hidden_size, num_layers=self.layer_num,
                            batch_first=True, dropout=drop_frac)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.output_size, bias=False)

        if initiallize:
            self.lstm.apply(self.init_weights)
            self.linear_act_name = 'tanh'
            self.fc_hidden.apply(self.init_weights)
            self.fc_encoder.apply(self.init_weights)
            self.linear_act_name = 'linear'
            self.linear.apply(self.init_weights)
            self.attention_weights.apply(self.init_weights)

    def forward(self, input, hidden, encoder_outputs):
        if hidden[0].size(0) == self.layer_num * 2:
            h_n, c_n = hidden
            h_n = h_n.reshape(self.layer_num, 2, h_n.size(1), self.hidden_size).mean(dim=1)
            c_n = c_n.reshape(self.layer_num, 2, c_n.size(1), self.hidden_size).mean(dim=1)
            hidden = (h_n.contiguous(), c_n.contiguous())

        temp_hidden = hidden[0][-1:]
        temp_hidden = torch.transpose(temp_hidden, 0, 1)
        alignment_score = torch.tanh(
            self.fc_hidden(temp_hidden) + self.fc_encoder(encoder_outputs))
        alignment_score = self.attention_weights(alignment_score)
        alignment_score = torch.transpose(alignment_score, 1, 2)
        applied_weights = F.softmax(alignment_score, dim=2)
        context_vector = torch.bmm(applied_weights, encoder_outputs)
        output = torch.cat((input, context_vector), dim=2)
        output, hidden = self.lstm(output, hidden)
        output = self.linear(output)
        return output, hidden, applied_weights.detach()

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.linear_act_name == 'relu' or self.linear_act_name == 'leaky_reLU':
                nn.init.kaiming_normal_(module.weight.detach(), mode='fan_out', nonlinearity=self.linear_act_name)
                # nn.init.kaiming_uniform_(module.weight.detach(), mode='fan_out', nonlinearity='relu')
            elif any(self.linear_act_name == np.array(['tanh', 'sigmoid', 'linear'])):
                nn.init.xavier_normal_(module.weight.detach(), gain=nn.init.calculate_gain(self.linear_act_name))
                # nn.init.xavier_uniform_(module.weight.detach(), gain=nn.init.calculate_gain(self.linear_act_name))
            else:
                raise NotImplementedError("Initializer got error!")
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
