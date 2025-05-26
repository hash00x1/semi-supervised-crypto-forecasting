
import torch
import torch.nn as nn


# Monte Carlo Attention Module
class MonteCarloAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.5):
        super(MonteCarloAttention, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)  # For bidirectional LSTM
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, lstm_output):
        attention_scores = self.attention(lstm_output)
        attention_scores = self.dropout(attention_scores)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.bmm(attention_weights.transpose(1, 2), lstm_output)
        return context_vector.squeeze(1)

# LSTM Classifier with Monte Carlo Attention
class SingleInputLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_units=128):
        super(SingleInputLSTMClassifier, self).__init__()
        self.hidden_units = hidden_units
        self.lstm1 = nn.LSTM(input_dim, hidden_units, batch_first=True, bidirectional=True)
        self.relu1 = nn.ReLU()
        self.lstm2 = nn.LSTM(hidden_units * 2, hidden_units, batch_first=True, bidirectional=True)
        self.relu2 = nn.ReLU()
        self.attention = MonteCarloAttention(hidden_units, dropout_rate=0.5)
        self.fc = nn.Linear(hidden_units * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(2, batch_size, self.hidden_units, device=x.device)
        c0 = torch.zeros(2, batch_size, self.hidden_units, device=x.device)
        out, _ = self.lstm1(x, (h0, c0))
        out = self.relu1(out)
        h1 = torch.zeros(2, batch_size, self.hidden_units, device=x.device)
        c1 = torch.zeros(2, batch_size, self.hidden_units, device=x.device)
        out, _ = self.lstm2(out, (h1, c1))
        out = self.relu2(out)
        context_vector = self.attention(out)
        out = self.fc(context_vector)
        out = self.sigmoid(out)
        return out