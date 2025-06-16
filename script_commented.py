# Import necessary libraries
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import librosa  # For audio processing
import torch
import torch.nn as nn
from transformers import HubertConfig, AutoConfig, AutoProcessor, HubertForSequenceClassification, Wav2Vec2Processor
import shutil
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
import argparse
from itertools import product

from script import test_speakers_combinations


def parse_args():
    """Parse command line arguments for training configuration"""
    parser = argparse.ArgumentParser(description="Train Hubert model with LoRA for speaker classification")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the audio data in filtered_folders")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store training results and models")
    parser.add_argument("--cache_dir", type=str, default='/home/spyridod/.cache/huggingface/datasets',
                        help="Directory for caching datasets")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to use for training (e.g., 'cuda', 'cpu')")
    parser.add_argument("--truncation", type=int, default=12,
                        help="Number of seconds to truncate each audio sample to (default: 12)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs to train for. ")
    return parser.parse_args()


# Define all speaker codes
all_speakers = ["M01", "M04", "M05", "M07", "M08", "M09", "M10", "F02", "F03", "F04", "F05", "M11", "M14", "M12", "M16"]

# Define excluded speakers (fixed for all iterations)
excluded_speakers = ["M01", "M09", "M10"]

# Define all speaker codes grouped by class (excluding excluded speakers)
speakers_by_class = {
    "VeryLow": [s for s in ["M04", "F03", "M12"] if s not in excluded_speakers],
    "Low": [s for s in ["M07", "F02", "M16"] if s not in excluded_speakers],
    "Medium": [s for s in ["M05", "F04", "M11"] if s not in excluded_speakers],
    "High": [s for s in ["M08", "F05", "M14"] if s not in excluded_speakers]
}

# Generate ALL possible combinations with one speaker from each class
all_possible_combinations = list(product(
    speakers_by_class["VeryLow"],
    speakers_by_class["Low"],
    speakers_by_class["Medium"],
    speakers_by_class["High"]
))

# Remove any combinations that might have duplicate speakers
unique_combinations = [list(combo) for combo in all_possible_combinations if len(set(combo)) == 4]

# This command is used for checkpointing
# Take just the first 10 combinations (starting from index 21 for some reason)
#test_speakers_combinations = unique_combinations[21:]
test_speakers_combinations = unique_combinations


#debugging
"""
print(f"Total possible combinations: {len(unique_combinations)}")
print("Selected first 10 combinations:")
for i, combo in enumerate(test_speakers_combinations, 1):
    print(f"Combination {i}: {combo}")
"""



# Define speaker to class mapping (which speaker belongs to which pitch class)
speaker_to_class = {
    "M01": "VeryLow",
    "M04": "VeryLow",
    "M05": "Medium",
    "M07": "Low",
    "M08": "High",
    "M09": "High",
    "M10": "High",
    "F02": "Low",
    "F03": "VeryLow",
    "F04": "Medium",
    "F05": "High",
    "M11": "Medium",
    "M14": "High",
    "M12": "VeryLow",
    "M16": "Low"
}

# Map class names to numerical labels
class_to_number = {"VeryLow": 0, "Low": 1, "Medium": 2, "High": 3}


class LoRALayer(nn.Module):
    """Implementation of Low-Rank Adaptation (LoRA) layer"""

    def __init__(self, in_dim, out_dim, rank=8, alpha=16, activation=True):
        super().__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.randn(in_dim, rank))  # Low-rank matrix A
        self.B = nn.Parameter(torch.zeros(rank, out_dim))  # Low-rank matrix B
        self.scale = alpha / rank  # Scaling factor
        self.non_linearity = nn.GELU()  # Activation function
        self.activation = activation  # Whether to use activation

    def forward(self, x):
        """Forward pass through the LoRA layer"""
        if self.activation:
            return self.non_linearity((x @ self.A @ self.B)) * self.scale
        else:
            return (x @ self.A @ self.B) * self.scale


class LinearWithLoRA(nn.Module):
    """Wrapper that combines a linear layer with a LoRA layer"""

    def __init__(self, linear_layer, rank=8, alpha=16, activation=True):
        super().__init__()
        self.linear = linear_layer  # Original linear layer
        self.lora = LoRALayer(linear_layer.in_features,
                              linear_layer.out_features,
                              rank, alpha, activation=activation)

    def forward(self, x):
        """Forward pass: original linear output + LoRA output"""
        return self.linear(x) + self.lora(x)


@dataclass
class DataCollatorCTCWithPadding:
    """Data collator that pads input sequences and labels"""
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, examples: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """Pad a batch of examples to the same length"""
        INPUT_FIELD = "input_values"
        LABEL_FIELD = "labels"
        input_features = [{INPUT_FIELD: example[INPUT_FIELD]} for example in examples]
        words = [example["words"] for example in examples]
        labels = [example[LABEL_FIELD] for example in examples]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch[LABEL_FIELD] = torch.tensor(labels)
        batch["words"] = words
        return batch


class HubertDataset(TorchDataset):
    """Dataset class for Hubert model"""

    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        item = self.dataset[idx]
        return {
            "input_values": torch.tensor(item["input_values"]),
            "labels": torch.tensor(class_to_number[item["label"]]),
            "speaker": item["speaker"],
            "words": item["words"]
        }


def detect_utterance_boundaries(audio, sr, top_db=25, frame_length=2048, hop_length=512):
    """Detect start and end of speech in an audio signal"""
    non_silent = librosa.effects.split(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length
    )
    if len(non_silent) == 0:
        return 0, len(audio)
    start = non_silent[0][0]
    end = non_silent[-1][1]
    return start, end


def process_utterance(path, target_sr=16000, truncation_seconds=12):
    """Process a single audio file:
    1. Load and resample
    2. Truncate to specified length
    3. Detect speech boundaries
    4. Add small buffers around speech segments"""
    audio, sr = librosa.load(path, sr=target_sr)
    max_samples = truncation_seconds * sr
    audio = audio[:max_samples]
    start_sample, end_sample = detect_utterance_boundaries(audio, sr)
    utterance = audio[start_sample:end_sample]
    buffer_samples = int(0.2 * sr)  # 200ms buffer
    utterance = np.concatenate([
        audio[max(0, start_sample - buffer_samples):start_sample],
        utterance,
        audio[end_sample:min(len(audio), end_sample + buffer_samples)]
    ])
    return utterance


def speech_file_to_array_fn(path, apply_normalization=False, truncation_seconds=12):
    """Convert audio file to numpy array with processing"""
    sr = 16000
    audio = process_utterance(path, target_sr=sr, truncation_seconds=truncation_seconds)
    return audio


def label_to_id(label, label_list):
    """Convert text label to numerical ID"""
    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1


def preprocess_function(examples, processor, label_list, input_column, output_column, truncation_seconds=12):
    """Preprocess a batch of examples:
    1. Convert audio files to arrays
    2. Extract labels
    3. Process with feature extractor"""
    speech_list = [speech_file_to_array_fn(path, truncation_seconds=truncation_seconds) for path in
                   examples[input_column]]
    target_list = [label_to_id(label, label_list) for label in examples[output_column]]

    result = processor(
        speech_list,
        sampling_rate=16000,
        padding=False
    )
    result["labels"] = list(target_list)
    result["words"] = [Path(path).stem for path in examples[input_column]]
    result["original_paths"] = examples[input_column]
    return result


def create_output_dir(output_base_dir, iteration, test_speakers):
    """Create output directory for current experiment"""
    test_speakers_str = "_".join(sorted(test_speakers))
    dir_name = f"{output_base_dir}/iteration_{iteration + 21}_speakers_{test_speakers_str}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def save_confusion_matrix(y_true, y_pred, output_dir, epoch):
    """Generate and save confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_to_number.keys(),
                yticklabels=class_to_number.keys())
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{output_dir}/confusion_matrix_epoch_{epoch}.png")
    plt.close()


def train_epoch(model, loader, optimizer, criterion, device):
    """Train model for one epoch"""
    model.train()
    total_loss, total_correct = 0, 0
    for batch in tqdm(loader, desc="Training"):
        features = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(input_values=features).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), total_correct / len(loader.dataset)


def eval_epoch(model, loader, criterion, epoch, output_dir, device):
    """Evaluate model on validation set"""
    model.eval()
    total_loss, total_correct = 0, 0
    all_preds = []
    all_labels = []
    wrong_predictions = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            features = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            words = batch["words"]

            outputs = model(features).logits
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)

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
    return total_loss / len(loader), total_correct / len(loader.dataset), all_preds, all_labels


def process_data(data_dir, test_speakers, excluded_speakers):
    """Process audio data from directory and split into train/val/test sets"""
    data = []
    seen_combinations = set()

    # Collect all audio files and their metadata
    for path in tqdm(Path(data_dir).glob("**/*.wav")):
        parts = path.stem.split('_')
        if len(parts) >= 4:
            speaker = parts[0]
            word = parts[2]
            combination = f"{speaker}_{word}"

            # Only add new combinations or words containing "UW"
            if combination not in seen_combinations or "UW" in word:
                data.append({
                    "name": path.stem,
                    "path": str(path),
                    "speaker": speaker,
                    "word": word,
                    "label": speaker_to_class[speaker],
                    "combination": combination,
                    "has_uw": "UW" in word
                })
                if "UW" not in word:
                    seen_combinations.add(combination)

    df = pd.DataFrame(data)
    # Custom split that ensures test speakers are only in test set
    train_df, valid_df, test_df = custom_train_test_split(
        df,
        test_speakers=test_speakers,
        excluded_speakers=excluded_speakers,
        test_size=0.1,
        random_state=None,
        stratify_col="speaker"
    )
    return train_df, valid_df, test_df


def custom_train_test_split(df, test_speakers, excluded_speakers, test_size=0.2, random_state=None, stratify_col=None):
    """Custom train/test split that handles excluded speakers and ensures test speakers are only in test set"""
    # First extract test set based on specified test speakers
    test_df = df[df['speaker'].isin(test_speakers)].copy()
    remaining_df = df[~df['speaker'].isin(test_speakers + excluded_speakers)].copy()

    # Stratify by speaker if requested
    if stratify_col:
        stratify = remaining_df[stratify_col]
    else:
        stratify = None

    # Split remaining data into train and validation
    train_df, val_df = train_test_split(
        remaining_df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def main():
    """Main training function"""
    args = parse_args()
    data_dir = args.data_dir
    output_base_dir = args.output_dir
    cache_dir = args.cache_dir
    device = torch.device(args.device)
    truncation_seconds = args.truncation
    num_epochs = args.epochs

    os.makedirs(output_base_dir, exist_ok=True)

    # Iterate through each speaker combination
    for i, test_speakers in enumerate(test_speakers_combinations):
        # Clear cache before each iteration
        for filename in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

        print(f"\nStarting iteration {i + 1} with test speakers: {test_speakers}")
        output_dir = create_output_dir(output_base_dir, i + 1, test_speakers)

        # Process and split data
        train_df, valid_df, test_df = process_data(data_dir, test_speakers, excluded_speakers)

        # Save data splits to CSV
        train_df.to_csv(f"{output_dir}/train_filtered_uw.csv", sep="\t", index=False)
        test_df.to_csv(f"{output_dir}/test_filtered_uw.csv", sep="\t", index=False)
        valid_df.to_csv(f"{output_dir}/valid_filtered_uw.csv", sep="\t", index=False)

        # Load datasets
        data_files = {
            "train": f"{output_dir}/train_filtered_uw.csv",
            "validation": f"{output_dir}/test_filtered_uw.csv",
        }
        dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

        input_column = "path"
        output_column = "label"
        label_list = train_dataset.unique(output_column)
        label_list.sort()
        num_labels = len(label_list)

        # Load pretrained Hubert model and processor
        model_name_or_path = "facebook/hubert-large-ls960-ft"
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            label2id={label: i for i, label in enumerate(label_list)},
            id2label={i: label for i, label in enumerate(label_list)},
        )
        processor = AutoProcessor.from_pretrained(model_name_or_path)
        target_sampling_rate = processor.feature_extractor.sampling_rate

        model = HubertForSequenceClassification.from_pretrained(model_name_or_path, config=config)

        # Apply LoRA to specific layers
        k = 2  # Layer index to modify
        model.hubert.encoder.layers[k].attention.q_proj = LinearWithLoRA(
            model.hubert.encoder.layers[k].attention.q_proj, rank=8, alpha=8)
        model.hubert.encoder.layers[k].attention.v_proj = LinearWithLoRA(
            model.hubert.encoder.layers[k].attention.v_proj, rank=8, alpha=8)

        # Freeze all parameters except LoRA ones
        for param in model.hubert.parameters():
            param.requires_grad = False

        # Unfreeze LoRA parameters
        model.hubert.encoder.layers[k].attention.q_proj.lora.A.requires_grad = True
        model.hubert.encoder.layers[k].attention.q_proj.lora.B.requires_grad = True
        model.hubert.encoder.layers[k].attention.v_proj.lora.A.requires_grad = True
        model.hubert.encoder.layers[k].attention.v_proj.lora.B.requires_grad = True

        # Preprocess datasets
        train_dataset = train_dataset.map(
            lambda x: preprocess_function(x, input_column=input_column, output_column=output_column,
                                          processor=processor, label_list=label_list,
                                          truncation_seconds=truncation_seconds),
            batch_size=100,
            batched=True,
            num_proc=4
        )
        eval_dataset = eval_dataset.map(
            lambda x: preprocess_function(x, input_column=input_column, output_column=output_column,
                                          processor=processor, label_list=label_list,
                                          truncation_seconds=truncation_seconds),
            batch_size=100,
            batched=True,
            num_proc=4
        )

        # Create data loaders
        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
        train_loader = DataLoader(
            HubertDataset(train_dataset),
            batch_size=8,
            collate_fn=data_collator,
            shuffle=True
        )
        eval_loader = DataLoader(
            HubertDataset(eval_dataset),
            collate_fn=data_collator,
            batch_size=8
        )

        # Set up training
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=5e-5)
        criterion = nn.CrossEntropyLoss()

        best_accuracy = 0
        best_train_accuracy = 0
        best_epoch = 0

        # Training loop
        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, preds, labels = eval_epoch(model, eval_loader, criterion, epoch + 1, output_dir, device)

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc * 100:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc * 100:.2f}%")

            save_confusion_matrix(labels, preds, output_dir, epoch + 1)

            # Save best model
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_train_accuracy = train_acc
                model_name = f"best_hubert_classifier_{'_'.join(sorted(test_speakers))}.pt"
                torch.save(model.state_dict(), f"{output_dir}/{model_name}")
                best_epoch = epoch

        # Save final model and training summary
        torch.save(model.state_dict(), f"{output_dir}/final_model.pt")
        with open(f"{output_dir}/training_summary.txt", "w") as f:
            f.write(f"Best validation accuracy: {best_accuracy * 100:.2f}%\n")
            f.write(f"Test speakers: {test_speakers}\n")
            f.write(f"Excluded speakers: {excluded_speakers}\n")
            f.write(f"Training samples: {len(train_dataset)}\n")
            f.write(f"Validation samples: {len(eval_dataset)}\n")
            f.write(f"Epoch of best model {best_epoch}\n")
            f.write(f"Training accuracy of best model {best_train_accuracy}")

    print("All iterations completed!")


if __name__ == "__main__":
    main()