"""
Speaker configuration file containing:
- All speaker IDs
- Excluded speakers
- Speaker to class mappings
- Class to numerical label mappings
"""

# All available speaker IDs in the dataset
all_speakers = ["M01", "M04", "M05", "M07", "M08", "M09", "M10", "F02", "F03", "F04", "F05", "M11", "M14", "M12", "M16"]

# Speakers to exclude (2 male speakers from High and one from Very Low)
excluded_speakers = ["M01", "M09", "M10"]

# Speakers grouped by their intelligibility classes (VeryLow, Low, Medium, High)
speakers_by_class = {
    "VeryLow": [s for s in ["M04", "F03", "M12"] if s not in excluded_speakers],
    "Low": [s for s in ["M07", "F02", "M16"] if s not in excluded_speakers],
    "Medium": [s for s in ["M05", "F04", "M11"] if s not in excluded_speakers],
    "High": [s for s in ["M08", "F05", "M14"] if s not in excluded_speakers]
}

# Mapping from speaker IDs to their intelligibility classes
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

# Mapping from class names to numerical labels
class_to_number = {"VeryLow": 0, "Low": 1, "Medium": 2, "High": 3}