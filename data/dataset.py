"""
Dataset classes and data collators for Hubert model training
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from torch.utils.data import Dataset as TorchDataset
from transformers import Wav2Vec2Processor
import torch

from config.speakers import class_to_number


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        processor: Wav2Vec2 processor containing feature extractor and tokenizer
        padding: Padding strategy
        max_length: Maximum length to pad/truncate to
        pad_to_multiple_of: Pad to multiple of this value
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, examples: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """Pad a batch of audio inputs and labels to equal length"""
        INPUT_FIELD = "input_values"
        LABEL_FIELD = "labels"

        # Extract input features and labels
        input_features = [{INPUT_FIELD: example[INPUT_FIELD]} for example in examples]
        words = [example["words"] for example in examples]
        labels = [example[LABEL_FIELD] for example in examples]

        # Pad inputs using huggingface processor
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Add labels and words to batch
        batch[LABEL_FIELD] = torch.tensor(labels)
        batch["words"] = words
        return batch


class HubertDataset(TorchDataset):
    """PyTorch Dataset for Hubert"""

    def __init__(self, hf_dataset):
        """
        Args:
            hf_dataset: HuggingFace dataset containing processed audio samples
        """
        self.dataset = hf_dataset

    def __len__(self):
        """Return number of samples in dataset"""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get single sample from dataset

        Returns:
            Dictionary containing:
            - input_values: audio features
            - labels: numerical class label
            - speaker: speaker ID
            - words: word transcription
        """
        item = self.dataset[idx]
        return {
            "input_values": torch.tensor(item["input_values"]),
            "labels": torch.tensor(class_to_number[item["label"]]),
            "speaker": item["speaker"],
            "words": item["words"]
        }