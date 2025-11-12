"""
RNN model architectures for sentiment classification
"""
import torch
import torch.nn as nn


class SentimentRNN(nn.Module):
    """
    Base RNN model for sentiment classification
    """
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=64,
                 num_layers=2, dropout=0.4, activation='relu', bidirectional=False):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden layers
            num_layers: Number of RNN layers
            dropout: Dropout rate
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            bidirectional: Whether to use bidirectional RNN
        """
        super(SentimentRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # RNN layer
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Fully connected output layer
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, 1)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, seq_length)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)

        # RNN
        rnn_out, hidden = self.rnn(embedded)  # rnn_out: (batch_size, seq_length, hidden_dim * num_directions)

        # Use the last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states from last layer
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]  # (batch_size, hidden_dim)

        # Apply activation and dropout
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)

        # Fully connected layer
        output = self.fc(hidden)  # (batch_size, 1)
        output = self.output_activation(output)

        return output


class SentimentLSTM(nn.Module):
    """
    LSTM model for sentiment classification
    """
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=64,
                 num_layers=2, dropout=0.4, activation='relu', bidirectional=False):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden layers
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            bidirectional: Whether to use bidirectional LSTM
        """
        super(SentimentLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Fully connected output layer
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, 1)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, seq_length)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Use the last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states from last layer
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]  # (batch_size, hidden_dim)

        # Apply activation and dropout
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)

        # Fully connected layer
        output = self.fc(hidden)  # (batch_size, 1)
        output = self.output_activation(output)

        return output


def create_model(model_type, vocab_size, embedding_dim=100, hidden_dim=64,
                 num_layers=2, dropout=0.4, activation='relu'):
    """
    Factory function to create models

    Args:
        model_type: Type of model ('rnn', 'lstm', 'bilstm')
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_dim: Dimension of hidden layers
        num_layers: Number of layers
        dropout: Dropout rate
        activation: Activation function ('relu', 'tanh', 'sigmoid')

    Returns:
        Model instance
    """
    if model_type.lower() == 'rnn':
        return SentimentRNN(
            vocab_size, embedding_dim, hidden_dim,
            num_layers, dropout, activation, bidirectional=False
        )
    elif model_type.lower() == 'lstm':
        return SentimentLSTM(
            vocab_size, embedding_dim, hidden_dim,
            num_layers, dropout, activation, bidirectional=False
        )
    elif model_type.lower() == 'bilstm':
        return SentimentLSTM(
            vocab_size, embedding_dim, hidden_dim,
            num_layers, dropout, activation, bidirectional=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation
    vocab_size = 10000

    print("Testing RNN model:")
    rnn_model = create_model('rnn', vocab_size)
    print(rnn_model)
    print(f"Parameters: {sum(p.numel() for p in rnn_model.parameters())}")

    print("\nTesting LSTM model:")
    lstm_model = create_model('lstm', vocab_size)
    print(lstm_model)
    print(f"Parameters: {sum(p.numel() for p in lstm_model.parameters())}")

    print("\nTesting Bidirectional LSTM model:")
    bilstm_model = create_model('bilstm', vocab_size)
    print(bilstm_model)
    print(f"Parameters: {sum(p.numel() for p in bilstm_model.parameters())}")

    # Test forward pass
    batch_size = 32
    seq_length = 50
    x = torch.randint(0, vocab_size, (batch_size, seq_length))

    output = rnn_model(x)
    print(f"\nRNN output shape: {output.shape}")

    output = lstm_model(x)
    print(f"LSTM output shape: {output.shape}")

    output = bilstm_model(x)
    print(f"BiLSTM output shape: {output.shape}")
