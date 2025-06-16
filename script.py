import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torchaudio
from sklearn.model_selection import train_test_split
import os
import sys
import warnings
from typing import Optional, Union, Tuple
import librosa
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import HubertConfig, AutoConfig, AutoProcessor, HubertForSequenceClassification, Wav2Vec2Processor
from transformers.integrations import is_deepspeed_zero3_enabled, is_fsdp_managed_module
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from transformers.models.hubert.modeling_hubert import HubertFeedForward, HUBERT_ATTENTION_CLASSES, \
    HubertPositionalConvEmbedding, _compute_mask_indices, HubertEncoder, HubertPreTrainedModel, HubertFeatureEncoder, \
    HubertFeatureProjection, HUBERT_INPUTS_DOCSTRING, _CONFIG_FOR_DOC, _HIDDEN_STATES_START_POSITION, \
    _SEQ_CLASS_EXPECTED_OUTPUT, _SEQ_CLASS_EXPECTED_LOSS, _SEQ_CLASS_CHECKPOINT
from transformers.utils import replace_return_docstrings, add_start_docstrings_to_model_forward, \
    add_code_sample_docstrings
import random
import shutil
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from torch.utils.data import Dataset as TorchDataset
from transformers import DataCollatorWithPadding, AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
import argparse
from itertools import product

def parse_args():
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

# Take just the first 10 combinations
test_speakers_combinations = unique_combinations[21:]

print(f"Total possible combinations: {len(unique_combinations)}")
print("Selected first 10 combinations:")
for i, combo in enumerate(test_speakers_combinations, 1):
    print(f"Combination {i}: {combo}")

# Define speaker to class mapping
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

class_to_number = {"VeryLow": 0, "Low": 1, "Medium": 2, "High": 3}


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16, activation=True):
        super().__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.randn(in_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scale = alpha / rank
        self.non_linearity = nn.GELU()
        self.activation = activation

    def forward(self, x):
        if self.activation:
            return self.non_linearity((x @ self.A @ self.B)) * self.scale
        else:
            return (x @ self.A @ self.B) * self.scale


class LinearWithLoRA(nn.Module):
    def __init__(self, linear_layer, rank=8, alpha=16, activation=True):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(linear_layer.in_features,
                              linear_layer.out_features,
                              rank, alpha, activation=activation)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, examples: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
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
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_values": torch.tensor(item["input_values"]),
            "labels": torch.tensor(class_to_number[item["label"]]),
            "speaker": item["speaker"],
            "words": item["words"]
        }


def detect_utterance_boundaries(audio, sr, top_db=25, frame_length=2048, hop_length=512):
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
    audio, sr = librosa.load(path, sr=target_sr)
    max_samples = truncation_seconds * sr
    audio = audio[:max_samples]
    start_sample, end_sample = detect_utterance_boundaries(audio, sr)
    utterance = audio[start_sample:end_sample]
    buffer_samples = int(0.2 * sr)
    utterance = np.concatenate([
        audio[max(0, start_sample - buffer_samples):start_sample],
        utterance,
        audio[end_sample:min(len(audio), end_sample + buffer_samples)]
    ])
    return utterance


def speech_file_to_array_fn(path, apply_normalization=False, truncation_seconds=12):
    sr = 16000
    audio = process_utterance(path, target_sr=sr, truncation_seconds=truncation_seconds)
    return audio


def label_to_id(label, label_list):
    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1


def preprocess_function(examples,  processor, label_list, input_column, output_column, truncation_seconds=12):
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
    test_speakers_str = "_".join(sorted(test_speakers))
    dir_name = f"{output_base_dir}/iteration_{iteration + 21}_speakers_{test_speakers_str}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def save_confusion_matrix(y_true, y_pred, output_dir, epoch):
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

            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    wrong_predictions.append({
                        'word': words[i],
                        'true_label': labels[i].item(),
                        'predicted_label': predicted[i].item()
                    })

    wrong_df = pd.DataFrame(wrong_predictions)
    wrong_df.to_csv(f"{output_dir}/wrong_predictions_epoch_{epoch}.csv", index=False)
    return total_loss / len(loader), total_correct / len(loader.dataset), all_preds, all_labels


def process_data(data_dir, test_speakers, excluded_speakers):
    data = []
    seen_combinations = set()
    for path in tqdm(Path(data_dir).glob("**/*.wav")):
        parts = path.stem.split('_')
        if len(parts) >= 4:
            speaker = parts[0]
            word = parts[2]
            combination = f"{speaker}_{word}"

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
    test_df = df[df['speaker'].isin(test_speakers)].copy()
    remaining_df = df[~df['speaker'].isin(test_speakers + excluded_speakers)].copy()

    if stratify_col:
        stratify = remaining_df[stratify_col]
    else:
        stratify = None

    train_df, val_df = train_test_split(
        remaining_df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def main():
    args = parse_args()
    data_dir = args.data_dir
    output_base_dir = args.output_dir
    cache_dir = args.cache_dir
    device = torch.device(args.device)
    truncation_seconds = args.truncation
    num_epochs = args.epochs

    os.makedirs(output_base_dir, exist_ok=True)

    for i, test_speakers in enumerate(test_speakers_combinations):
        # Clear cache
        for filename in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

        print(f"\nStarting iteration {i + 1} with test speakers: {test_speakers}")
        output_dir = create_output_dir(output_base_dir, i + 1, test_speakers)

        train_df, valid_df, test_df = process_data(data_dir, test_speakers, excluded_speakers)

        train_df.to_csv(f"{output_dir}/train_filtered_uw.csv", sep="\t", index=False)
        test_df.to_csv(f"{output_dir}/test_filtered_uw.csv", sep="\t", index=False)
        valid_df.to_csv(f"{output_dir}/valid_filtered_uw.csv", sep="\t", index=False)

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

        k = 2
        model.hubert.encoder.layers[k].attention.q_proj = LinearWithLoRA(
            model.hubert.encoder.layers[k].attention.q_proj, rank=8, alpha=8)
        model.hubert.encoder.layers[k].attention.v_proj = LinearWithLoRA(
            model.hubert.encoder.layers[k].attention.v_proj, rank=8, alpha=8)

        for param in model.hubert.parameters():
            param.requires_grad = False

        model.hubert.encoder.layers[k].attention.q_proj.lora.A.requires_grad = True
        model.hubert.encoder.layers[k].attention.q_proj.lora.B.requires_grad = True
        model.hubert.encoder.layers[k].attention.v_proj.lora.A.requires_grad = True
        model.hubert.encoder.layers[k].attention.v_proj.lora.B.requires_grad = True

        train_dataset = train_dataset.map(
            lambda x: preprocess_function(x, input_column=input_column, output_column=output_column, processor=processor, label_list=label_list,
                                          truncation_seconds=truncation_seconds),
            batch_size=100,
            batched=True,
            num_proc=4
        )
        eval_dataset = eval_dataset.map(
            lambda x: preprocess_function(x,input_column=input_column, output_column=output_column, processor=processor, label_list=label_list,
                                          truncation_seconds=truncation_seconds),
            batch_size=100,
            batched=True,
            num_proc=4
        )

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

        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=5e-5)
        criterion = nn.CrossEntropyLoss()

        best_accuracy = 0
        best_train_accuracy = 0
        best_epoch = 0
        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, preds, labels = eval_epoch(model, eval_loader, criterion, epoch + 1, output_dir, device)

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc * 100:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc * 100:.2f}%")

            save_confusion_matrix(labels, preds, output_dir, epoch + 1)

            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_train_accuracy = train_acc
                model_name = f"best_hubert_classifier_{'_'.join(sorted(test_speakers))}.pt"
                torch.save(model.state_dict(), f"{output_dir}/{model_name}")
                best_epoch = epoch

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