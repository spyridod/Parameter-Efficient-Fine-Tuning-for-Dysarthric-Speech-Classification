"""
Training and evaluation utilities
"""

from tqdm import tqdm
import torch
import pandas as pd


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Train model for one epoch

    Args:
        model: The model
        loader: DataLoader for training data
        optimizer: Optimization algorithm
        criterion: Loss function
        device: Device to train on

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    total_loss, total_correct = 0, 0

    for batch in tqdm(loader, desc="Training"):
        # Move batch to device
        features = batch["input_values"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_values=features).logits
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == labels).sum().item()

    # Return average loss and accuracy
    return total_loss / len(loader), total_correct / len(loader.dataset)


def eval_epoch(model, loader, criterion, epoch, output_dir, device):
    """
    Evaluate model on validation (test set)

    Args:
        model: The model
        loader: DataLoader for validation(test) data
        criterion: Loss function
        epoch: Current epoch number (for logging)
        output_dir: Directory to save wrong predictions
        device: Device to evaluate on

    Returns:
        Tuple of (average loss, accuracy, predictions, labels)
    """
    model.eval()
    total_loss, total_correct = 0, 0
    all_preds = []
    all_labels = []
    wrong_predictions = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            # Move batch to device
            features = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            words = batch["words"]

            # Forward pass
            outputs = model(features).logits
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)

            # Accumulate metrics
            total_loss += loss.item()
            total_correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Record wrong predictions for analysis
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    wrong_predictions.append({
                        'word': words[i],
                        'true_label': labels[i].item(),
                        'predicted_label': predicted[i].item()
                    })

    # Save wrong predictions to CSV
    wrong_df = pd.DataFrame(wrong_predictions)
    wrong_df.to_csv(f"{output_dir}/wrong_predictions_epoch_{epoch}.csv", index=False)

    return (
        total_loss / len(loader),
        total_correct / len(loader.dataset),
        all_preds,
        all_labels
    )