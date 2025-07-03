#!/usr/bin/env python3
"""
1. Processes multiple experiment folders containing trained models
2. Evaluates each model on its corresponding test set
3. Generates comprehensive metrics including accuracy, confusion matrices, ROC curves
4. Produces aggregated reports across all iterations

Usage:
    python evaluate_models.py --folder1 <path1> --folder2 <path2> --output_dir <output_path>
"""

import itertools
import os
from math import sqrt
import argparse
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
import librosa
from transformers import Wav2Vec2Processor, HubertConfig, HubertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List, Dict, Union, Optional
from scipy.stats import norm
import torch.nn as nn

# ======================== Helper Functions ========================

def roc_auc_ci(y_true, y_score, positive=1):
    """
    Calculate confidence interval for ROC AUC score using Hanley-McNeil method.

    Args:
        y_true: True binary labels
        y_score: Target scores
        positive: Label of the positive class (default=1)

    Returns:
        Tuple of (lower_bound, upper_bound) for 95% confidence interval
    """
    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2 * AUC ** 2 / (1 + AUC)
    SE_AUC = sqrt((AUC * (1 - AUC) + (N1 - 1) * (Q1 - AUC ** 2) + (N2 - 1) * (Q2 - AUC ** 2)) / (N1 * N2))
    lower = AUC - 1.96 * SE_AUC
    upper = AUC + 1.96 * SE_AUC
    return (max(0, lower), min(1, upper))


def find_optimal_roc_point(fpr, tpr):
    """
    Find the optimal operating point on ROC curve using Youden's J statistic.

    Args:
        fpr: Array of false positive rates
        tpr: Array of true positive rates

    Returns:
        Tuple of (optimal_fpr, optimal_tpr) and optimal index
    """
    j_scores = tpr - fpr  # Calculate Youden's J statistic
    optimal_idx = np.argmax(j_scores)
    return (fpr[optimal_idx], tpr[optimal_idx]), optimal_idx


def average_confusion_matrices(all_cms, class_names):
    """
    Average multiple confusion matrices and normalize.

    Args:
        all_cms: List of confusion matrices
        class_names: List of class names for labeling

    Returns:
        Normalized average confusion matrix
    """
    if not all_cms:
        return None

    avg_cm = np.sum(all_cms, axis=0)
    with np.errstate(all='ignore'):
        avg_cm = avg_cm.astype('float') / avg_cm.sum(axis=1)[:, np.newaxis]
        return np.nan_to_num(avg_cm)


def plot_confusion_matrix(cm, class_names, output_path):
    """
    Plot and save a confusion matrix.

    Args:
        cm: Confusion matrix to plot
        class_names: List of class names
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_scores_labels(y_scores, y_true, output_path):
    """
    Save prediction scores and true labels

    Args:
        y_scores: Model prediction scores
        y_true: True labels
        output_path: Base path for saving files
    """
    np.savez(output_path, scores=y_scores, labels=y_true)

    # Save as CSV for readability
    scores_df = pd.DataFrame(y_scores, columns=[f"class_{i}" for i in range(y_scores.shape[1])])
    scores_df['true_label'] = y_true
    scores_df.to_csv(output_path.replace('.npz', '.csv'), index=False)


# ======================== Evaluation Functions ========================

def calculate_roc_curve(y_scores, y_true):
    """
    Calculate ROC curve for multi-class classification using One-vs-One approach.

    Args:
        y_scores: Model scores (logits) for each class
        y_true: True class labels

    Returns:
        fpr_grid: Common FPR grid points
        average_tpr: Average TPR values
        mean_auc: Mean AUC across all class pairs
        mean_ci_lower: Mean lower CI bound
        mean_ci_upper: Mean upper CI bound
    """
    roc_curves = []
    num_classes = len(np.unique(y_true))
    pairs = list(itertools.combinations(range(num_classes), 2))

    for class_1, class_2 in pairs:
        # Create binary classification problem for this pair
        mask = (y_true == class_1) | (y_true == class_2)
        y_true_pair = y_true[mask]
        y_scores_pair = y_scores[mask]
        y_binary = (y_true_pair == class_2).astype(int)
        y_score = y_scores_pair[:, class_2] - y_scores_pair[:, class_1]

        # Calculate ROC metrics
        fpr, tpr, _ = roc_curve(y_binary, y_score)
        roc_auc = auc(fpr, tpr)

        try:
            ci_lower, ci_upper = roc_auc_ci(y_binary, y_score)
        except:
            ci_lower, ci_upper = (0, 1)

        # Handle edge cases
        if len(fpr) < 2 or len(tpr) < 2:
            fpr = np.array([0, 1])
            tpr = np.array([0, 1])
            roc_auc = 0.5
            ci_lower, ci_upper = (0, 1)

        roc_curves.append({
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'class_pair': (class_1, class_2)
        })

    # Interpolate all curves to common FPR grid
    fpr_grid = np.linspace(0, 1, 100)
    interpolated_tprs = []
    auc_scores = []
    ci_lowers = []
    ci_uppers = []

    for curve in roc_curves:
        fpr = curve['fpr']
        tpr = curve['tpr']

        # Remove duplicates while preserving order
        unique_fpr, unique_indices = np.unique(fpr, return_index=True)
        unique_tpr = tpr[unique_indices]

        # Interpolate to common grid
        interp_tpr = interp1d(unique_fpr, unique_tpr, kind='linear',
                              bounds_error=False, fill_value=(0, 1))
        tpr_interpolated = np.nan_to_num(interp_tpr(fpr_grid), nan=0.0)

        interpolated_tprs.append(tpr_interpolated)
        auc_scores.append(curve['auc'])
        ci_lowers.append(curve['ci_lower'])
        ci_uppers.append(curve['ci_upper'])

    # Calculate averages
    if not interpolated_tprs:
        return np.linspace(0, 1, 100), np.linspace(0, 1, 100), 0.5, 0.0, 1.0

    average_tpr = np.mean(interpolated_tprs, axis=0)
    mean_auc = np.mean(auc_scores)
    mean_ci_lower = np.mean(ci_lowers)
    mean_ci_upper = np.mean(ci_uppers)

    return fpr_grid, average_tpr, mean_auc, mean_ci_lower, mean_ci_upper


def plot_average_roc_curve(all_fpr, all_tpr, all_auc, all_ci_lower, all_ci_upper, output_dir):
    """
    Plot average ROC curve with confidence intervals.

    Args:
        all_fpr: List of FPR arrays from all iterations
        all_tpr: List of TPR arrays from all iterations
        all_auc: List of AUC values from all iterations
        all_ci_lower: List of lower CI bounds
        all_ci_upper: List of upper CI bounds
        output_dir: Directory to save the plot

    Returns:
        Dictionary with optimal point information
    """
    plt.figure(figsize=(8, 6))

    # Calculate averages
    avg_fpr = np.mean(all_fpr, axis=0)
    avg_tpr = np.mean(all_tpr, axis=0)
    mean_auc = np.mean(all_auc)
    mean_ci_lower = np.mean(all_ci_lower)
    mean_ci_upper = np.mean(all_ci_upper)

    # Find optimal operating point
    optimal_point, optimal_idx = find_optimal_roc_point(avg_fpr, avg_tpr)

    # Plot confidence interval
    plt.fill_between(avg_fpr,
                     [tpr - 0.02 for tpr in avg_tpr],
                     [tpr + 0.02 for tpr in avg_tpr],
                     color='blue', alpha=0.2,
                     label=f'95% CI: [{mean_ci_lower:.2f}, {mean_ci_upper:.2f}]')

    # Plot main curve and optimal point
    plt.plot(avg_fpr, avg_tpr, color='blue', linewidth=2,
             label=f'Average ROC (AUC = {mean_auc:.2f})')
    plt.scatter(optimal_point[0], optimal_point[1], color='red', marker='o',
                label=f'Optimal point (FPR={optimal_point[0]:.2f}, TPR={optimal_point[1]:.2f})')

    # Format plot
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve with 95% CI')
    plt.legend(loc="lower right")
    plt.grid(which='both', linestyle='--', linewidth=0.5)

    # Save plot
    plt.savefig(os.path.join(output_dir, 'average_roc_curve_with_ci.png'), format="png")
    plt.close()

    return {
        'optimal_fpr': optimal_point[0],
        'optimal_tpr': optimal_point[1],
        'optimal_idx': optimal_idx,
        'mean_auc': mean_auc,
        'ci_lower': mean_ci_lower,
        'ci_upper': mean_ci_upper
    }


# ======================== Data Classes and Model Components ========================

@dataclass
class DataCollatorCTCWithPadding:
    """Data collator for Hubert model with dynamic padding."""
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, examples: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """Collate and pad batch of examples."""
        INPUT_FIELD = "input_values"
        LABEL_FIELD = "labels"

        input_features = [{INPUT_FIELD: example[INPUT_FIELD]} for example in examples]
        words = [example["words"] for example in examples]
        labels = [example[LABEL_FIELD] for example in examples]
        speaker = [example["speaker"] for example in examples]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch[LABEL_FIELD] = torch.tensor(labels)
        batch["words"] = words
        batch["speaker"] = speaker
        return batch


class HubertDataset(Dataset):
    """Dataset class for Hubert"""

    def __init__(self, df, processor):
        self.df = df
        self.processor = processor
        self.label_to_id = {"VeryLow": 0, "Low": 1, "Medium": 2, "High": 3}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Load and process a single audio sample."""
        item = self.df.iloc[idx]
        audio = self.process_utterance(item['path'])
        label = self.label_to_id[item['label']]
        return {
            "input_values": audio,
            "labels": label,
            "words": item['word'],
            "speaker": item['speaker']
        }

    def detect_utterance_boundaries(self, audio, sr, top_db=25, frame_length=2048, hop_length=512):
        """Detect non-silent segments in audio."""
        non_silent = librosa.effects.split(
            audio,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )
        return (0, len(audio)) if len(non_silent) == 0 else (non_silent[0][0], non_silent[-1][1])

    def process_utterance(self, path, target_sr=16000):
        """Process audio file into model inputs."""
        audio, sr = librosa.load(path, sr=target_sr)
        audio = audio[:16000 * 12]  # Truncate to 12 seconds
        start_sample, end_sample = self.detect_utterance_boundaries(audio, sr)

        # Add buffer around utterance
        buffer_samples = int(0.2 * sr)
        utterance = np.concatenate([
            audio[max(0, start_sample - buffer_samples):start_sample],
            audio[start_sample:end_sample],
            audio[end_sample:min(len(audio), end_sample + buffer_samples)]
        ])

        # Process through feature extractor
        inputs = self.processor(
            utterance,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True
        )
        return inputs.input_values[0]


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer

    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        rank: Rank of the low-rank matrices
        alpha: Scaling factor
        activation: Whether to apply GELU activation (Modified non-linear LoRA)
    """

    def __init__(self, in_dim, out_dim, rank=8, alpha=16, activation=True):
        super().__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.randn(in_dim, rank))  # Low-rank matrix A
        self.B = nn.Parameter(torch.zeros(rank, out_dim))  # Low-rank matrix B
        self.scale = alpha / rank  # Scaling factor
        self.non_linearity = nn.GELU()  # Activation function
        self.activation = activation  # Whether to use activation (Modified non-linear LoRA)

    def forward(self, x):
        """
        Forward pass: BAx*scale with optional activation
        B and A are reversed
        """
        if self.activation:
            return self.non_linearity((x @ (self.B.transpose(0,1) @ self.A.transpose(0,1)))) * self.scale
        else:
            return (x @ (self.B.transpose(0,1) @ self.A.transpose(0,1))) * self.scale


class LinearWithLoRA(nn.Module):
    """
    Wrapper that combines a linear layer with a LoRA layer

    Args:
        linear_layer: Original linear layer to augment
        rank: Rank of LoRA matrices
        alpha: Scaling factor
        activation: Whether LoRA uses activation (LoRA or Modified NL LoRA)
    """

    def __init__(self, linear_layer, rank=8, alpha=16, activation=True):
        super().__init__()
        self.linear = linear_layer  # Original linear layer
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank, alpha,
            activation=activation
        )

    def forward(self, x):
        """Forward pass: original output + LoRA output"""
        return self.linear(x) + self.lora(x)

# ======================== Model Loading and Evaluation ========================

def load_model_from_folder(folder_path, device='cuda'):
    """Load trained model from experiment folder."""
    # Find model file
    model_files = [f for f in os.listdir(folder_path)
                   if f.startswith('best_hubert_classifier') and f.endswith('.pt')]
    if not model_files:
        raise FileNotFoundError(f"No model file found in {folder_path}")

    # Load or create config
    config_path = os.path.join(folder_path, 'config.json')
    if os.path.exists(config_path):
        config = HubertConfig.from_pretrained(folder_path)
    else:
        config = HubertConfig.from_pretrained("facebook/hubert-large-ls960-ft")
        config.num_labels = 4
        config.label2id = {"VeryLow": 0, "Low": 1, "Medium": 2, "High": 3}
        config.id2label = {0: "VeryLow", 1: "Low", 2: "Medium", 3: "High"}

    # Initialize model with LoRA layers
    model = HubertForSequenceClassification(config)
    k = 2  # Layer to apply LoRA to (matches training setup)
    model.hubert.encoder.layers[k].attention.q_proj = LinearWithLoRA(
        model.hubert.encoder.layers[k].attention.q_proj, rank=8, alpha=8)
    model.hubert.encoder.layers[k].attention.v_proj = LinearWithLoRA(
        model.hubert.encoder.layers[k].attention.v_proj, rank=8, alpha=8)

    # Load weights and prepare for evaluation
    model_path = os.path.join(folder_path, model_files[0])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def create_test_loader(test_df, processor):
    """Create test DataLoader with proper preprocessing."""
    dataset = HubertDataset(test_df, processor)
    collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    return DataLoader(
        dataset,
        batch_size=8,
        collate_fn=collator,
        shuffle=False
    )


def evaluate_model(model, test_loader, device='cuda', iteration_dir=None):
    """Evaluate model on test set and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_speakers = []
    all_scores = []
    wrong_predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move batch to device
            features = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            words = batch["words"]
            speakers = batch["speaker"]

            # Get model predictions
            outputs = model(features).logits
            _, predicted = torch.max(outputs, 1)

            # Store results
            all_scores.append(outputs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_speakers.extend(speakers)

            # Record wrong predictions
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    wrong_predictions.append({
                        'word': words[i],
                        'speaker': speakers[i],
                        'true_label': labels[i].item(),
                        'predicted_label': predicted[i].item()
                    })

    # Calculate metrics
    y_scores = np.concatenate(all_scores)
    y_true = np.array(all_labels)

    if iteration_dir:
        save_scores_labels(y_scores, y_true, os.path.join(iteration_dir, 'scores_labels.npz'))

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds,
                               target_names=["VeryLow", "Low", "Medium", "High"],
                               output_dict=True)

    # Calculate ROC curve
    fpr, tpr, roc_auc, ci_lower, ci_upper = calculate_roc_curve(y_scores, y_true)

    # Calculate per-speaker accuracy
    speaker_acc = {}
    df = pd.DataFrame({'speaker': all_speakers, 'correct': (np.array(all_preds) == np.array(all_labels))})
    for speaker, group in df.groupby('speaker'):
        speaker_acc[speaker] = group['correct'].mean()

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': cr,
        'speaker_accuracy': speaker_acc,
        'wrong_predictions': wrong_predictions,
        'all_predictions': list(zip(all_speakers, all_labels, all_preds)),
        'roc_curve': (fpr, tpr, roc_auc, ci_lower, ci_upper),
    }


# ======================== Main Processing Functions ========================

def process_single_iteration(iteration_folder, output_dir, device):
    """Process evaluation for a single iteration folder."""
    # Setup paths and directories
    folder_name = os.path.basename(iteration_folder)
    iteration_output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(iteration_output_dir, exist_ok=True)

    # Initialize components
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = load_model_from_folder(iteration_folder, device)

    # Load and prepare test data
    test_csv = os.path.join(iteration_folder, 'test_filtered_uw.csv')
    if not os.path.exists(test_csv):
        print(f"Test CSV not found in {iteration_folder}")
        return None

    test_df = pd.read_csv(test_csv, sep='\t')
    test_loader = create_test_loader(test_df, processor)

    # Run evaluation
    metrics = evaluate_model(model, test_loader, device, iteration_output_dir)

    # Save iteration results
    if metrics:
        # Save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=["VeryLow", "Low", "Medium", "High"],
                    yticklabels=["VeryLow", "Low", "Medium", "High"])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(iteration_output_dir, 'confusion_matrix.png'))
        plt.close()

        # Save wrong predictions
        wrong_df = pd.DataFrame(metrics['wrong_predictions'])
        wrong_df.to_csv(os.path.join(iteration_output_dir, 'wrong_predictions.csv'), index=False)

        # Save speaker accuracy
        speaker_acc_df = pd.DataFrame.from_dict(metrics['speaker_accuracy'], orient='index', columns=['accuracy'])
        speaker_acc_df.to_csv(os.path.join(iteration_output_dir, 'speaker_accuracy.csv'))

        # Save metrics report
        with open(os.path.join(iteration_output_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(
                [x[1] for x in metrics['all_predictions']],
                [x[2] for x in metrics['all_predictions']],
                target_names=["VeryLow", "Low", "Medium", "High"]
            ))

    return metrics


def create_comprehensive_reports(all_metrics, output_dir):
    """Generate aggregated reports across all iterations."""
    summary_data = []
    all_fpr = []
    all_tpr = []
    all_auc = []
    all_ci_lower = []
    all_ci_upper = []
    all_cms = []
    class_names = ["VeryLow", "Low", "Medium", "High"]

    # Process metrics from all iterations
    for metrics in all_metrics:
        if not metrics:
            continue

        cr = metrics['classification_report']
        roc_data = metrics.get('roc_curve', ([], [], 0.0, 0.0, 0.0))

        # Store confusion matrix
        if metrics['confusion_matrix'] is not None:
            all_cms.append(metrics['confusion_matrix'])

        # Collect summary statistics
        summary_data.append({
            'iteration': metrics.get('iteration', 0),
            'test_speakers': '_'.join(metrics.get('test_speakers', [])),
            'accuracy': metrics['accuracy'],
            'VeryLow_precision': cr['VeryLow']['precision'],
            'VeryLow_recall': cr['VeryLow']['recall'],
            'VeryLow_f1': cr['VeryLow']['f1-score'],
            'Low_precision': cr['Low']['precision'],
            'Low_recall': cr['Low']['recall'],
            'Low_f1': cr['Low']['f1-score'],
            'Medium_precision': cr['Medium']['precision'],
            'Medium_recall': cr['Medium']['recall'],
            'Medium_f1': cr['Medium']['f1-score'],
            'High_precision': cr['High']['precision'],
            'High_recall': cr['High']['recall'],
            'High_f1': cr['High']['f1-score'],
            'macro_avg_f1': cr['macro avg']['f1-score'],
            'weighted_avg_f1': cr['weighted avg']['f1-score'],
            'roc_auc': roc_data[2]
        })

        # Collect ROC curve data
        if len(roc_data[0]) > 0 and len(roc_data[1]) > 0:
            all_fpr.append(roc_data[0])
            all_tpr.append(roc_data[1])
            all_auc.append(roc_data[2])
            all_ci_lower.append(roc_data[3])
            all_ci_upper.append(roc_data[4])

    # Save average confusion matrix
    if all_cms:
        avg_cm = average_confusion_matrices(all_cms, class_names)
        if avg_cm is not None:
            plot_confusion_matrix(
                avg_cm,
                class_names,
                os.path.join(output_dir, 'average_confusion_matrix.png')
            )
            np.savetxt(
                os.path.join(output_dir, 'average_confusion_matrix.csv'),
                avg_cm,
                delimiter=',',
                fmt='%.4f'
            )

    # Save average ROC curve
    roc_info = None
    if all_fpr and all_tpr:
        roc_info = plot_average_roc_curve(all_fpr, all_tpr, all_auc, all_ci_lower, all_ci_upper, output_dir)

    # Save summary statistics
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'summary_metrics.csv'), index=False)

    # Generate overall report
    with open(os.path.join(output_dir, 'overall_report.txt'), 'w') as f:
        f.write(f"Total iterations processed: {len(all_metrics)}\n")
        f.write(f"Average accuracy: {np.mean([m['accuracy'] for m in all_metrics if m]):.4f}\n")

        if all_auc:
            f.write(f"Average ROC AUC: {np.mean(all_auc):.4f}\n")
            f.write(f"Best ROC AUC: {np.max(all_auc):.4f}\n")
            f.write(f"Worst ROC AUC: {np.min(all_auc):.4f}\n")

        if roc_info:
            f.write(f"\nOptimal ROC operating point:\n")
            f.write(f"FPR: {roc_info['optimal_fpr']:.3f}\n")
            f.write(f"TPR: {roc_info['optimal_tpr']:.3f}\n")

        f.write("\nDetailed metrics per iteration:\n")
        for m in all_metrics:
            if not m:
                continue
            f.write(f"\nIteration {m.get('iteration', 'N/A')} (speakers: {'_'.join(m.get('test_speakers', []))})\n")
            f.write(f"Accuracy: {m['accuracy']:.4f}\n")
            f.write(f"Macro F1: {m['classification_report']['macro avg']['f1-score']:.4f}\n")
            if 'roc_curve' in m and m['roc_curve'][2]:
                f.write(f"ROC AUC: {m['roc_curve'][2]:.4f}\n")

    print(f"\nEvaluation complete. Results saved to {output_dir}")


def process_results_folders(folder1, folder2, output_dir):
    """Main function to process all experiment folders."""
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_metrics = []

    # Find all iteration folders
    iteration_folders = []
    for folder in [folder1, folder2]:
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} does not exist")
            continue
        for subdir in os.listdir(folder):
            if subdir.startswith('iteration_'):
                iteration_folders.append(os.path.join(folder, subdir))

    # Process each iteration
    for iteration_folder in tqdm(iteration_folders, desc="Processing iterations"):
        # Extract iteration info
        folder_name = os.path.basename(iteration_folder)
        parts = folder_name.split('_')
        iteration_num = int(parts[1])
        speakers = parts[3:]

        print(f"\nProcessing iteration {iteration_num} with speakers: {speakers}")
        metrics = process_single_iteration(iteration_folder, output_dir, device)

        if metrics:
            metrics.update({
                'iteration': iteration_num,
                'test_speakers': speakers
            })
            all_metrics.append(metrics)

    # Create comprehensive reports
    create_comprehensive_reports(all_metrics, output_dir)


# ======================== Main Script ========================

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate Hubert classification models.')
    parser.add_argument('--folder1', type=str, required=True,
                        help='Path to first experiment folder containing iteration subfolders')
    parser.add_argument('--folder2', type=str, default='',
                        help='Path to second experiment folder (optional)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation results')

    args = parser.parse_args()

    # Run evaluation
    process_results_folders(
        args.folder1,
        args.folder2 if args.folder2 else args.folder1,  # Use folder1 twice if folder2 not provided
        args.output_dir
    )