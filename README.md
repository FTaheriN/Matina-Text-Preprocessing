# Persian Text Preprocessing and Deduplication Pipeline

## Introduction
---
Processing Persian text presents unique challenges, and existing preprocessing pipelines often fail to comprehensively handle the variety of textual anomalies present in real-world datasets. Many available tools lack robustness in filtering noisy data, normalizing text, and effectively detecting duplicates. To address these issues, we introduce a comprehensive Persian text preprocessing and deduplication pipeline that ensures high-quality data processing tailored to the specific needs of Persian text corpora.

## Pipeline Overview
---
This pipeline is designed to clean, normalize, and deduplicate Persian text effectively. It applies a series of filtering and normalization techniques, including:
- Document filtering based on length, character composition, and stopword ratio.
- Text normalization, including Unicode standardization and date formatting.
- Removal of personal information and dataset-specific artifacts.
- Line-level filtering for symbols, numbers, and formatting inconsistencies.
- Document deduplication using Lean MinHashing.

## Customization
---
This pipeline is highly configurable. The filtering criteria, normalization rules, and deduplication thresholds can all be adjusted to suit different document types, contexts, and processing needs.

<!--## Paper and Dataset
---
 For a detailed explanation of the methodology and evaluation, please refer to our published paper:
- **Paper Title:** [Insert Paper Title]
- **Link:** [Insert Paper Link]

Additionally, the dataset processed by this pipeline is available on Hugging Face:
- **Hugging Face Repository:** [Insert Hugging Face Link] -->

## Installation and Usage
---
To use this pipeline, create a YAML configuration file setting the desired parameters (or use the default values). Sample configurations are available in the `config` directory.

Run the pipeline using:
```bash
python main.py {config file path}
```

