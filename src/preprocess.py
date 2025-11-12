"""
Data preprocessing for sentiment analysis
"""
import pandas as pd
import numpy as np
import re
import string
from collections import Counter
import pickle


class TextPreprocessor:
    """
    Preprocessor for text data
    """
    def __init__(self, max_words=10000):
        self.max_words = max_words
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

    def clean_text(self, text):
        """
        Clean text: lowercase and remove punctuation/special characters
        """
        # Lowercase
        text = text.lower()
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove punctuation and special characters (keep only alphanumeric and spaces)
        text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def tokenize(self, text):
        """
        Simple whitespace tokenization
        """
        return text.split()

    def build_vocabulary(self, texts):
        """
        Build vocabulary from texts, keeping only top max_words
        """
        all_words = []
        for text in texts:
            clean = self.clean_text(text)
            tokens = self.tokenize(clean)
            all_words.extend(tokens)

        # Count word frequencies
        word_counts = Counter(all_words)

        # Keep only top max_words
        most_common = word_counts.most_common(self.max_words - 2)  # Reserve 2 for PAD and UNK

        # Build word2idx and idx2word
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}

        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self.vocab_size = len(self.word2idx)
        print(f"Vocabulary size: {self.vocab_size}")

        return self.vocab_size

    def text_to_sequence(self, text):
        """
        Convert text to sequence of token IDs
        """
        clean = self.clean_text(text)
        tokens = self.tokenize(clean)
        sequence = [self.word2idx.get(token, 1) for token in tokens]  # 1 is UNK
        return sequence

    def texts_to_sequences(self, texts):
        """
        Convert multiple texts to sequences
        """
        sequences = [self.text_to_sequence(text) for text in texts]
        return sequences

    def pad_sequences(self, sequences, max_length, padding='post', truncating='post'):
        """
        Pad or truncate sequences to fixed length
        """
        padded = np.zeros((len(sequences), max_length), dtype=np.int32)

        for i, seq in enumerate(sequences):
            if len(seq) > max_length:
                # Truncate
                if truncating == 'post':
                    padded[i] = seq[:max_length]
                else:
                    padded[i] = seq[-max_length:]
            else:
                # Pad
                if padding == 'post':
                    padded[i, :len(seq)] = seq
                else:
                    padded[i, -len(seq):] = seq

        return padded

    def save_vocabulary(self, filepath):
        """
        Save vocabulary to file
        """
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size,
            'max_words': self.max_words
        }
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"Vocabulary saved to {filepath}")

    def load_vocabulary(self, filepath):
        """
        Load vocabulary from file
        """
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        self.word2idx = vocab_data['word2idx']
        self.idx2word = vocab_data['idx2word']
        self.vocab_size = vocab_data['vocab_size']
        self.max_words = vocab_data['max_words']
        print(f"Vocabulary loaded from {filepath}")


def load_and_preprocess_data(data_path, max_words=10000, seq_length=50, test_size=0.5):
    """
    Load and preprocess the IMDb dataset

    Args:
        data_path: Path to the dataset CSV file
        max_words: Maximum number of words to keep in vocabulary
        seq_length: Length to pad/truncate sequences
        test_size: Fraction of data to use for testing (0.5 for 50/50 split)

    Returns:
        X_train, X_test, y_train, y_test, preprocessor, stats
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Extract texts and labels
    texts = df['review'].values
    labels = (df['sentiment'] == 'positive').astype(int).values

    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(texts)}")
    print(f"Positive samples: {labels.sum()}")
    print(f"Negative samples: {len(labels) - labels.sum()}")

    # Split into train and test (50/50 split as per requirements)
    split_idx = int(len(texts) * (1 - test_size))
    train_texts = texts[:split_idx]
    test_texts = texts[split_idx:]
    y_train = labels[:split_idx]
    y_test = labels[split_idx:]

    print(f"\nTrain samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")

    # Initialize preprocessor
    preprocessor = TextPreprocessor(max_words=max_words)

    # Build vocabulary on training data
    print("\nBuilding vocabulary...")
    preprocessor.build_vocabulary(train_texts)

    # Convert texts to sequences
    print("Converting texts to sequences...")
    train_sequences = preprocessor.texts_to_sequences(train_texts)
    test_sequences = preprocessor.texts_to_sequences(test_texts)

    # Calculate average sequence length
    train_lengths = [len(seq) for seq in train_sequences]
    test_lengths = [len(seq) for seq in test_sequences]
    avg_train_len = np.mean(train_lengths)
    avg_test_len = np.mean(test_lengths)

    print(f"\nSequence Length Statistics:")
    print(f"Average train sequence length: {avg_train_len:.2f}")
    print(f"Average test sequence length: {avg_test_len:.2f}")
    print(f"Max train sequence length: {max(train_lengths)}")
    print(f"Max test sequence length: {max(test_lengths)}")

    # Pad sequences
    print(f"\nPadding sequences to length {seq_length}...")
    X_train = preprocessor.pad_sequences(train_sequences, seq_length)
    X_test = preprocessor.pad_sequences(test_sequences, seq_length)

    # Prepare statistics dictionary
    stats = {
        'total_samples': len(texts),
        'train_samples': len(train_texts),
        'test_samples': len(test_texts),
        'vocab_size': preprocessor.vocab_size,
        'avg_review_length': (avg_train_len + avg_test_len) / 2,
        'avg_train_length': avg_train_len,
        'avg_test_length': avg_test_len,
        'max_train_length': max(train_lengths),
        'max_test_length': max(test_lengths),
        'positive_samples': labels.sum(),
        'negative_samples': len(labels) - labels.sum()
    }

    print("\nPreprocessing complete!")
    return X_train, X_test, y_train, y_test, preprocessor, stats


if __name__ == "__main__":
    # Test preprocessing
    data_path = "../data/IMDB Dataset.csv"
    X_train, X_test, y_train, y_test, preprocessor, stats = load_and_preprocess_data(
        data_path, max_words=10000, seq_length=50
    )

    print(f"\nFinal shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

    # Save vocabulary
    preprocessor.save_vocabulary("../results/vocabulary.pkl")
