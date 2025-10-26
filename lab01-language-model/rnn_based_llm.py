import torch


class RnnBasedModel(torch.nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, rnn_hidden_dim: int, num_layers: int, drop_rate: float):
        super(RnnBasedModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_dim)
        self.rnn = torch.nn.LSTM(input_size=emb_dim, hidden_size=rnn_hidden_dim, num_layers=num_layers, dropout=drop_rate, batch_first=True)

        self.fc = torch.nn.Linear(rnn_hidden_dim, vocab_size)

        self.drop = torch.nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.drop(x)
        x = self.fc(x)
        # don't use softmax here because CrossEntropyLoss need logits.
        # x = torch.nn.Softmax(x, dim=-1)
        return x
