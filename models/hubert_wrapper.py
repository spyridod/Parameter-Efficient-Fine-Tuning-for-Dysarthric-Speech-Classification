"""
Wrapper for loading and configuring Hubert model
"""

from transformers import HubertForSequenceClassification, AutoConfig, AutoProcessor


def get_hubert_model(num_labels, label2id, id2label, model_name="facebook/hubert-large-ls960-ft"):
    """
    Load and configure pretrained Hubert model

    Args:
        num_labels: Number of output classes
        label2id: Mapping from label names to numerical IDs
        id2label: Mapping from numerical IDs to label names
        model_name: Pretrained model name

    Returns:
        Tuple of (model, processor)
    """
    # Load model config
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    # Load processor (feature extractor + tokenizer)
    processor = AutoProcessor.from_pretrained(model_name)

    # Load model with classification head
    model = HubertForSequenceClassification.from_pretrained(model_name, config=config)

    return model, processor