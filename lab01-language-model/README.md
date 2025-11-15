# Lab 01: Language Modeling (+ Lab 02 Extensions)

This directory contains implementations for training Polish language models from scratch. The code was initially developed for Lab 01 and later extended to support Lab 02 requirements without unnecessary duplication.

## Overview

### Lab 01: Language Modeling
Goal: Train language models that generate sentences in Polish using two different architectures:
- **RNN** (LSTM-based architecture)
- **Transformer** (GPT-2-like decoder-only architecture)

[Lab 01 Instructions](https://github.com/apohllo/computational-linguistics/blob/main/1-language-modeling.md)

### Lab 02: Tokenization (Extensions)
Goal: Train custom tokenizers and compare training results across three different tokenization methods:
- Pre-trained tokenizer
- SentencePiece tokenizer
- Whitespace tokenizer

[Lab 02 Instructions](https://github.com/apohllo/computational-linguistics/blob/main/2-tokenization.md)

**Note:** This codebase was extended to support Lab 02 requirements by adding tokenization capabilities, allowing the same implementation to be used for both laboratories.

## Data Source

Polish language data is retrieved from [speakleash](https://github.com/speakleash/speakleash).

## Key Extensions for Lab 02

- Added `custom_tokenizers.py` implementing classes for training whitespace and SentencePiece tokenizers
- Adjusted `llm_train.py` script to support different tokenizer types
- Created script to retrieve data from Speakleash and save as TXT/JSONL files
- Modified solution to use previously saved datasets (TXT/JSONL files) instead of retrieving from Speakleash API each time
- This improvement was necessary because retrieving data from the Speakleash API requires downloading the file manifest each time, even when the dataset is already downloaded to disk

## File Structure

- `dataset.py` - PyTorch dataloader for Speakleash data
- `transformer_based_llm.py` - GPT-2-like architecture implemented in PyTorch
- `rnn_based_llm.py` - RNN architecture implemented in PyTorch
- `models.py` - Pydantic models for storing model, dataset, and training parameters
- `utils.py` - Helper functions for encoding/decoding, loss calculation, etc.
- `llm_train.py` - Main training script with configurable parameters. Most parameters are set by manually editing `MODEL_CONFIG`, `TRAINING_SETTINGS`, or `DATASET_SETTINGS`. Additional parameters are handled via `argparse`
- `transformer_inference.py` - Script to test transformer inference efficiency and output quality
- `rnn_inference.py` - Script to test RNN inference efficiency and output quality
- `custom_tokenizers.py` - Implementation of custom whitespace and SentencePiece tokenizers

## Usage

### Example Execution
```bash
python llm_train.py --model_type transformer --tokenizer speakleash/Bielik-4.5B-v3 --dataset_name plwiki --tokenizer_type transformers --max_training_minutes 120
```

### Configurable Parameters

- `model_type` - Architecture type: `transformer` or `rnn`
- `tokenizer` - Tokenizer name or path
- `tokenizer_type` - Tokenizer type: `transformers`, `sentencepiece`, or `custom` (whitespace)
- `dataset_name` - Name of dataset from Speakleash
- `max_training_minutes` - Maximum training duration in minutes

## Results

A detailed report containing in-depth information and results for Lab 01 can be found in [report.md](report.md).