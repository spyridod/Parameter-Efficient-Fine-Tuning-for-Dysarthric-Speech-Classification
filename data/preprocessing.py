import librosa
import numpy as np
from pathlib import Path


def detect_utterance_boundaries(audio, sr, top_db=25, frame_length=2048, hop_length=512):
    """
    Detect start and end samples of speech in audio using energy threshold

    Args:
        audio: numpy array of audio samples
        sr: sample rate
        top_db: threshold in decibels
        frame_length: STFT frame length
        hop_length: STFT hop length

    Returns:
        Tuple of (start_sample, end_sample)
    """
    non_silent = librosa.effects.split(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length
    )
    if len(non_silent) == 0:  # If no speech is detected, use entire audio
        return 0, len(audio)
    start = non_silent[0][0]
    end = non_silent[-1][1]
    return start, end


def process_utterance(path, target_sr=16000, truncation_seconds=12):
    """
    Process single audio file:
    1. Load audio
    2. Truncate to max length
    3. Detect speech boundaries
    4. Add small buffers around speech

    Args:
        path: Path to audio file
        target_sr: Target sample rate
        truncation_seconds: Max length in seconds

    Returns:
        Processed audio array
    """
    # Load audio
    audio, sr = librosa.load(path, sr=target_sr)

    # Truncate to specified length
    max_samples = truncation_seconds * sr
    audio = audio[:max_samples]

    # Detect speech boundaries
    start_sample, end_sample = detect_utterance_boundaries(audio, sr)
    utterance = audio[start_sample:end_sample]

    # Add 200ms buffers around speech
    buffer_samples = int(0.2 * sr)
    utterance = np.concatenate([
        audio[max(0, start_sample - buffer_samples):start_sample],
        utterance,
        audio[end_sample:min(len(audio), end_sample + buffer_samples)]
    ])
    return utterance


def speech_file_to_array_fn(path, apply_normalization=False, truncation_seconds=12):
    """
    Convert audio file to numpy array with processing

    Args:
        path: Path to audio file
        truncation_seconds: Max length in seconds

    Returns:
        Processed audio array at 16kHz
    """
    sr = 16000  # Hubert expects 16kHz audio
    audio = process_utterance(path, target_sr=sr, truncation_seconds=truncation_seconds)
    return audio


def label_to_id(label, label_list):
    """
    Convert text label to numerical ID

    Args:
        label: Text label
        label_list: List of possible labels

    Returns:
        Numerical ID or -1 if label not found
    """
    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1


def preprocess_function(examples, processor, label_list, input_column, output_column, truncation_seconds=12):
    """
    Preprocess batch of examples for Hubert model

    Args:
        examples: Batch of examples
        processor: Hubert feature processor
        label_list: List of possible labels
        input_column: Name of column containing audio paths
        output_column: Name of column containing labels
        truncation_seconds: Max audio length

    Returns:
        Dictionary of processed features
    """
    # Convert audio files to arrays
    speech_list = [speech_file_to_array_fn(path, truncation_seconds=truncation_seconds)
                   for path in examples[input_column]]

    # Convert labels to IDs
    target_list = [label_to_id(label, label_list) for label in examples[output_column]]

    # Extract features using processor
    result = processor(
        speech_list,
        sampling_rate=16000,
        padding=False
    )

    # Add additional metadata
    result["labels"] = list(target_list)
    result["words"] = [Path(path).stem for path in examples[input_column]]
    result["original_paths"] = examples[input_column]
    return result