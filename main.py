
import argparse
import os
from itertools import product

import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import Adam

# Import from our modules
from config.speakers import *
from config.defaults import *
from data.dataset import DataCollatorCTCWithPadding, HubertDataset
from data.preprocessing import preprocess_function
from data.splitting import process_data
from models.lora import LinearWithLoRA
from models.hubert_wrapper import get_hubert_model
from training.train_utils import train_epoch, eval_epoch
from training.metrics import save_confusion_matrix
from utils.helpers import create_output_dir, clear_cache


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train Hubert model with LoRA for Speech Intelligibility classification"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the audio data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store training results and models"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="Directory for caching datasets"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Device to use for training (e.g., 'cuda', 'cpu')"
    )
    parser.add_argument(
        "--truncation",
        type=int,
        default=DEFAULT_TRUNCATION,
        help="Number of seconds to truncate each audio sample to"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of epochs to train for"
    )
    return parser.parse_args()


def main():
    """Main training loop"""
    args = parse_args()
    data_dir = args.data_dir
    output_base_dir = args.output_dir
    cache_dir = args.cache_dir
    device = torch.device(args.device)
    truncation_seconds = args.truncation
    num_epochs = args.epochs

    # Create output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)

    # Generate all possible speaker combinations for testing
    all_possible_combinations = list(product(
        speakers_by_class["VeryLow"],
        speakers_by_class["Low"],
        speakers_by_class["Medium"],
        speakers_by_class["High"]
    ))
    # Remove combinations with duplicate speakers
    unique_combinations = [list(combo) for combo in all_possible_combinations if len(set(combo)) == 4]

    # Here for checkpointing
    # Use combinations starting from index 21 (for some experimental reason)
    test_speakers_combinations = unique_combinations[21:]


    test_speakers_combinations = unique_combinations

    # Train model for each speaker combination
    for i, test_speakers in enumerate(test_speakers_combinations):
        # Clear cache before each iteration
        clear_cache(cache_dir)

        print(f"\nStarting iteration {i + 1} with test speakers: {test_speakers}")

        # Create output directory for this iteration
        output_dir = create_output_dir(output_base_dir, i + 1, test_speakers)

        # Process and split data
        train_df, valid_df, test_df = process_data(data_dir, test_speakers, excluded_speakers)

        # Save data splits to CSV
        train_df.to_csv(f"{output_dir}/train_filtered_uw.csv", sep="\t", index=False)
        test_df.to_csv(f"{output_dir}/test_filtered_uw.csv", sep="\t", index=False)
        valid_df.to_csv(f"{output_dir}/valid_filtered_uw.csv", sep="\t", index=False)

        # Load datasets using HuggingFace datasets
        data_files = {
            "train": f"{output_dir}/train_filtered_uw.csv",
            "validation": f"{output_dir}/test_filtered_uw.csv",
        }
        dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

        # Prepare model configuration
        input_column = "path"
        output_column = "label"
        label_list = train_dataset.unique(output_column)
        label_list.sort()
        num_labels = len(label_list)

        # Load Hubert model and processor
        model, processor = get_hubert_model(
            num_labels,
            {label: i for i, label in enumerate(label_list)},
            {i: label for i, label in enumerate(label_list)}
        )

        # Apply LoRA to specific attention layers
        k = 2  # Layer index to modify
        model.hubert.encoder.layers[k].attention.q_proj = LinearWithLoRA(
            model.hubert.encoder.layers[k].attention.q_proj,
            rank=8,
            alpha=8
        )
        model.hubert.encoder.layers[k].attention.v_proj = LinearWithLoRA(
            model.hubert.encoder.layers[k].attention.v_proj,
            rank=8,
            alpha=8
        )

        # Freeze all Hubert parameters except LoRA ones
        for param in model.hubert.parameters():
            param.requires_grad = False

        # Unfreeze LoRA parameters
        model.hubert.encoder.layers[k].attention.q_proj.lora.A.requires_grad = True
        model.hubert.encoder.layers[k].attention.q_proj.lora.B.requires_grad = True
        model.hubert.encoder.layers[k].attention.v_proj.lora.A.requires_grad = True
        model.hubert.encoder.layers[k].attention.v_proj.lora.B.requires_grad = True

        # Preprocess datasets
        train_dataset = train_dataset.map(
            lambda x: preprocess_function(
                x,
                processor,
                label_list,
                input_column,
                output_column,
                truncation_seconds
            ),
            batch_size=100,
            batched=True,
            num_proc=4
        )
        eval_dataset = eval_dataset.map(
            lambda x: preprocess_function(
                x,
                processor,
                label_list,
                input_column,
                output_column,
                truncation_seconds
            ),
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

        # Training setup
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=5e-5)
        criterion = nn.CrossEntropyLoss()

        # Track best model
        best_accuracy = 0
        best_train_accuracy = 0
        best_epoch = 0

        # Training loop
        for epoch in range(num_epochs):
            # Train and evaluate
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, preds, labels = eval_epoch(
                model, eval_loader, criterion, epoch + 1, output_dir, device
            )

            # Print metrics
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc * 100:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc * 100:.2f}%")

            # Save confusion matrix
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