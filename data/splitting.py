import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm

from config.speakers import speaker_to_class


def process_data(data_dir, test_speakers, excluded_speakers):
    """
    Process audio data from directory and create DataFrame

    Args:
        data_dir: Directory containing audio files
        test_speakers: List of speaker IDs to reserve for test set
        excluded_speakers: List of speaker IDs to exclude

    Returns:
        Tuple of (train_df, valid_df, test_df) DataFrames
    """
    data = []
    seen_combinations = set()  # Track seen speaker-word combinations

    # Collect all audio files
    for path in tqdm(Path(data_dir).glob("**/*.wav")):
        parts = path.stem.split('_')
        if len(parts) >= 4:  # Ensure proper filename format
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
                # Only track non-UW words to avoid duplicates (we keep only unique utterances)
                if "UW" not in word:
                    seen_combinations.add(combination)

    # Create DataFrame and split
    df = pd.DataFrame(data)
    return custom_train_test_split(
        df,
        test_speakers=test_speakers,
        excluded_speakers=excluded_speakers,
        test_size=0.1,
        random_state=None,
        stratify_col="speaker"
    )


def custom_train_test_split(df, test_speakers, excluded_speakers, test_size=0.2, random_state=None, stratify_col=None):
    """
    Custom train/test split that handles:
    - Reserved test speakers
    - Excluded speakers
    - Stratification according to speakers

    Args:
        df: Input DataFrame
        test_speakers: Speakers to reserve for test set
        excluded_speakers: Speakers to exclude
        test_size: Fraction of remaining data for validation (is not used)
        random_state: Random seed
        stratify_col: Column to stratify by

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # First extract test set based on specified test speakers
    test_df = df[df['speaker'].isin(test_speakers)].copy()

    # Remaining data (excluding test and excluded speakers)
    remaining_df = df[~df['speaker'].isin(test_speakers + excluded_speakers)].copy()

    # Prepare stratification if requested
    stratify = remaining_df[stratify_col] if stratify_col else None

    # Split remaining into train and validation
    train_df, val_df = train_test_split(
        remaining_df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)