"""
Training script for sentiment classification models
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from utils import Timer
import time


def create_optimizer(model, optimizer_name, learning_rate=0.001):
    """
    Create optimizer based on name

    Args:
        model: The model to optimize
        optimizer_name: Name of optimizer ('adam', 'sgd', 'rmsprop')
        learning_rate: Learning rate

    Returns:
        Optimizer instance
    """
    if optimizer_name.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name.lower() == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def train_epoch(model, train_loader, criterion, optimizer, device, gradient_clipping=False, clip_value=1.0):
    """
    Train for one epoch

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        gradient_clipping: Whether to apply gradient clipping
        clip_value: Gradient clipping value

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).float().unsqueeze(1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_x)

        # Calculate loss
        loss = criterion(outputs, batch_y)

        # Backward pass
        loss.backward()

        # Gradient clipping if enabled
        if gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        # Update weights
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate model on a dataset

    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Average loss, accuracy, predictions, true labels
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float().unsqueeze(1)

            # Forward pass
            outputs = model(batch_x)

            # Calculate loss
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            num_batches += 1

            # Get predictions
            predictions = (outputs >= 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / num_batches
    all_predictions = np.array(all_predictions).flatten()
    all_labels = np.array(all_labels).flatten()

    accuracy = (all_predictions == all_labels).mean()

    return avg_loss, accuracy, all_predictions, all_labels


def train_model(model, X_train, y_train, X_val, y_val, config, device):
    """
    Train the model with the given configuration

    Args:
        model: The model to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Training configuration dictionary
        device: Device to train on

    Returns:
        Dictionary with training history and metrics
    """
    # Configuration
    batch_size = config.get('batch_size', 32)
    epochs = config.get('epochs', 10)
    optimizer_name = config.get('optimizer', 'adam')
    learning_rate = config.get('learning_rate', 0.001)
    gradient_clipping = config.get('gradient_clipping', False)
    clip_value = config.get('clip_value', 1.0)

    # Create data loaders
    train_dataset = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.LongTensor(X_val), torch.LongTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create optimizer and loss function
    optimizer = create_optimizer(model, optimizer_name, learning_rate)
    criterion = nn.BCELoss()

    # Move model to device
    model = model.to(device)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'epoch_times': []
    }

    print(f"\nTraining configuration:")
    print(f"  Optimizer: {optimizer_name}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Gradient clipping: {gradient_clipping}")
    if gradient_clipping:
        print(f"  Clip value: {clip_value}")

    # Training loop
    print(f"\nStarting training...")
    for epoch in range(epochs):
        epoch_timer = Timer()
        epoch_timer.start()

        # Train for one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device,
                                 gradient_clipping, clip_value)

        # Evaluate on training and validation sets
        _, train_acc, _, _ = evaluate(model, train_loader, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        # Record epoch time
        epoch_time = epoch_timer.stop()

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(epoch_time)

        # Print progress
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"Time: {epoch_time:.2f}s")

    # Calculate average epoch time
    avg_epoch_time = np.mean(history['epoch_times'])
    print(f"\nTraining complete! Average epoch time: {avg_epoch_time:.2f}s")

    return history, avg_epoch_time


if __name__ == "__main__":
    from preprocess import load_and_preprocess_data
    from models import create_model
    from utils import set_seed, get_device

    # Set seed for reproducibility
    set_seed(42)

    # Get device
    device = get_device()

    # Load and preprocess data
    data_path = "../data/IMDB Dataset.csv"
    X_train, X_test, y_train, y_test, preprocessor, stats = load_and_preprocess_data(
        data_path, max_words=10000, seq_length=50, test_size=0.5
    )

    # Create model
    model = create_model('lstm', preprocessor.vocab_size)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")

    # Training configuration
    config = {
        'batch_size': 32,
        'epochs': 5,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'gradient_clipping': False,
        'clip_value': 1.0
    }

    # Train model
    history, avg_epoch_time = train_model(model, X_train, y_train, X_test, y_test, config, device)

    print("\nTraining history:")
    print(f"Final train accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")
