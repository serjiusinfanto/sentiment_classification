"""
Evaluation script for sentiment classification models
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report


def evaluate_model(model, X_test, y_test, device, batch_size=32):
    """
    Evaluate model and compute metrics

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        device: Device to evaluate on
        batch_size: Batch size for evaluation

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    model = model.to(device)

    # Create data loader
    test_dataset = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)

            # Forward pass
            outputs = model(batch_x)

            # Get predictions
            probabilities = outputs.cpu().numpy()
            predictions = (outputs >= 0.5).float().cpu().numpy()

            all_probabilities.extend(probabilities)
            all_predictions.extend(predictions)
            all_labels.extend(batch_y.numpy())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions).flatten()
    all_labels = np.array(all_labels).flatten()
    all_probabilities = np.array(all_probabilities).flatten()

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    f1_binary = f1_score(all_labels, all_predictions, average='binary')
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_binary': f1_binary,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }

    return metrics


def print_evaluation_results(metrics):
    """
    Print evaluation results in a formatted way

    Args:
        metrics: Dictionary of metrics from evaluate_model
    """
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"F1-Score (Binary): {metrics['f1_binary']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("="*50)


def compare_models(results_list, model_names):
    """
    Compare multiple models

    Args:
        results_list: List of metric dictionaries
        model_names: List of model names

    Returns:
        Comparison dataframe
    """
    import pandas as pd

    comparison_data = []
    for i, (results, name) in enumerate(zip(results_list, model_names)):
        comparison_data.append({
            'Model': name,
            'Accuracy': results['accuracy'],
            'F1-Score (Macro)': results['f1_macro'],
            'F1-Score (Binary)': results['f1_binary'],
            'Precision': results['precision'],
            'Recall': results['recall']
        })

    df = pd.DataFrame(comparison_data)
    return df


if __name__ == "__main__":
    from preprocess import load_and_preprocess_data
    from models import create_model
    from train import train_model
    from utils import set_seed, get_device

    # Set seed for reproducibility
    set_seed(42)

    # Get device
    device = get_device()

    # Load and preprocess data
    print("Loading and preprocessing data...")
    data_path = "../data/IMDB Dataset.csv"
    X_train, X_test, y_train, y_test, preprocessor, stats = load_and_preprocess_data(
        data_path, max_words=10000, seq_length=50, test_size=0.5
    )

    # Create and train a model
    print("\nCreating and training LSTM model...")
    model = create_model('lstm', preprocessor.vocab_size)

    config = {
        'batch_size': 32,
        'epochs': 3,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'gradient_clipping': False
    }

    history, avg_epoch_time = train_model(model, X_train, y_train, X_test, y_test, config, device)

    # Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test, y_test, device)
    print_evaluation_results(metrics)

    # Print classification report
    print("\nDetailed Classification Report:")
    print(classification_report(metrics['labels'], metrics['predictions'],
                                target_names=['Negative', 'Positive']))
