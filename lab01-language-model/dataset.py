import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from speakleash import Speakleash
import os
from speakleash.dataset import SpeakleashDataset
from typing import Optional, List

class PlWikiDataset(Dataset):
    def __init__(self, plwiki: SpeakleashDataset, tokenizer, max_length, stride, max_docs, split="train", train_ratio=0.8, val_ratio=0.1):
        """
        Dataset for Polish Wikipedia data with train/validation/test splits.

        Args:
            plwiki: PlWiki dataset object from Speakleash
            tokenizer: Tokenizer to encode text
            max_length: Maximum sequence length
            stride: Stride for sliding window
            max_docs: Maximum number of documents to process
            split: 'train', 'val', or 'test'
            train_ratio: Ratio of data for training (default: 0.8)
            val_ratio: Ratio of data for validation (default: 0.1)
                      test_ratio will be 1 - train_ratio - val_ratio
        """
        self.plwiki = plwiki
        self.plwiki_data = self.plwiki.data
        self.max_docs = min(max_docs, self.plwiki.manifest["stats"]["documents"])

        # Calculate split indices
        train_end = int(self.max_docs * train_ratio)
        val_end = int(self.max_docs * (train_ratio + val_ratio))

        if split == "train":
            doc_range = range(0, train_end)
        elif split == "val":
            doc_range = range(train_end, val_end)
        elif split == "test":
            doc_range = range(val_end, self.max_docs)
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        self.input_ids = []
        self.target_ids = []

        # Process documents for the specified split with error handling
        processed_docs = 0
        skipped_docs = 0
        current_doc_index = 0

        for doc in self.plwiki_data:
            # Check if we've reached our max_docs limit
            if current_doc_index >= self.max_docs:
                break

            # Check if this document is in our split range
            if current_doc_index in doc_range:
                try:
                    # Tokenize the current document
                    token_ids = tokenizer.encode(doc, allowed_special={"<|endoftext|>"})

                    # Skip very short documents
                    if len(token_ids) <= max_length:
                        current_doc_index += 1
                        continue

                    # Use a sliding window to chunk the document into overlapping sequences of max_length
                    for j in range(0, len(token_ids) - max_length, stride):
                        input_chunk = token_ids[j : j + max_length]
                        target_chunk = token_ids[j + 1 : j + max_length + 1]
                        self.input_ids.append(torch.tensor(input_chunk))
                        self.target_ids.append(torch.tensor(target_chunk))

                    processed_docs += 1

                except Exception as e:
                    print(f"Skipping corrupted document at index {current_doc_index}: {e}")
                    skipped_docs += 1

            current_doc_index += 1

        print(f"Successfully processed {processed_docs} documents for '{split}' split, skipped {skipped_docs} corrupted documents")
        print(f"Generated {len(self.input_ids)} sequences")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_plwiki_dataloader(
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    split="train",
    max_docs=100,
    train_ratio=0.8,
    val_ratio=0.1,
):
    """
    Create a dataloader for Polish Wikipedia data with train/validation/test splits.

    Args:
        batch_size: Batch size for the dataloader
        max_length: Maximum sequence length
        stride: Stride for sliding window
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop the last incomplete batch
        num_workers: Number of worker processes for data loading
        split: 'train', 'val', or 'test'
        max_docs: Maximum number of documents to process
        train_ratio: Ratio of data for training (default: 0.8)
        val_ratio: Ratio of data for validation (default: 0.1)

    Returns:
        DataLoader for the specified split
    """
    # Initialize Speakleash and get plwiki dataset
    base_dir = os.path.join(os.path.dirname(__file__))
    replicate_to = os.path.join(base_dir, "datasets")
    speakleash = Speakleash(replicate_to)
    plwiki = speakleash.get("plwiki")

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = PlWikiDataset(plwiki, tokenizer, max_length, stride, max_docs, split=split, train_ratio=train_ratio, val_ratio=val_ratio)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader


def get_plwiki_dataloader(
    split="train", batch_size=4, max_length=256, stride=128, max_docs=100, train_ratio=0.8, val_ratio=0.1, num_workers=0
):
    """
    Get a specific dataloader (train, val, or test).

    Args:
        split: Which split to get ('train', 'val', or 'test')
        batch_size: Batch size for the dataloader
        max_length: Maximum sequence length
        stride: Stride for sliding window
        max_docs: Maximum number of documents to process
        train_ratio: Ratio of data for training (default: 0.8)
        val_ratio: Ratio of data for validation (default: 0.1)
        num_workers: Number of worker processes for data loading

    Returns:
        DataLoader for the specified split
    """
    # Set appropriate shuffle and drop_last based on split
    if split == "train":
        shuffle = True
        drop_last = True
    else:  # val or test
        shuffle = False
        drop_last = False

    return create_plwiki_dataloader(
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        split=split,
        max_docs=max_docs,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )


def create_all_plwiki_dataloaders(batch_size=4, max_length=256, stride=128, max_docs=100, train_ratio=0.8, val_ratio=0.1, num_workers=0):
    """
    Create train, validation, and test dataloaders (if you really need all at once).

    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader)
    """
    train_dataloader = get_plwiki_dataloader("train", batch_size, max_length, stride, max_docs, train_ratio, val_ratio, num_workers)
    val_dataloader = get_plwiki_dataloader("val", batch_size, max_length, stride, max_docs, train_ratio, val_ratio, num_workers)
    test_dataloader = get_plwiki_dataloader("test", batch_size, max_length, stride, max_docs, train_ratio, val_ratio, num_workers)

    return train_dataloader, val_dataloader, test_dataloader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader