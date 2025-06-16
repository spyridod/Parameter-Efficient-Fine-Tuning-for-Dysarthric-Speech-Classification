"""
Metrics and visualization utilities
"""

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from config.speakers import class_to_number


def save_confusion_matrix(y_true, y_pred, output_dir, epoch):
    """
    Generate and save confusion matrix plot

    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_dir: Directory to save plot
        epoch: Epoch number (for filename)
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_to_number.keys(),
                yticklabels=class_to_number.keys())
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Save and close
    plt.savefig(f"{output_dir}/confusion_matrix_epoch_{epoch}.png")
    plt.close()