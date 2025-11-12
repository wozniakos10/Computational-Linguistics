from pathlib import Path

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Literal
from utils import load_split_data
from logger import get_configured_logger

logger = get_configured_logger(__name__)


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, add_special_tokens=False)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


class SpeakleashDataLoader(Dataset):
    def __init__(
        self,
        tokenizer,
        max_length,
        stride,
        speakleash_dataset_name,
        split="train",
        dataset_dir="train_datasets",
        file_type: Literal["jsonl", "txt"] = "txt",
    ):
        """
        Dataset for Polish Wikipedia data with train/validation/test splits.

        Args:
            plwiki: PlWiki dataset object from Speakleash
            tokenizer: Tokenizer to encode text
            max_length: Maximum sequence length
            stride: Stride for sliding window
            split: 'train', 'val', or 'test'
            train_ratio: Ratio of data for training (default: 0.8)
            val_ratio: Ratio of data for validation (default: 0.1)
                      test_ratio will be 1 - train_ratio - val_ratio
        """

        self.input_ids = []
        self.target_ids = []
        docs_list = []

        file_path = Path(dataset_dir) / speakleash_dataset_name
        text_data = load_split_data(data_dir=file_path, split=split, file_type=file_type)

        if file_type == "jsonl":
            # Here's the is assumption that I process each document seperately
            # maybe it would be better to concataenate all documents and add special tokens between them
            # like <|end_of_text|> to signal the end of one document and the start of another
            for doc in docs_list:
                # not allowing llamatokenizer to add token BOS. It adds it by default
                token_ids = tokenizer.encode(doc, add_special_tokens=False)

                # Use a sliding window to chunk the document into overlapping sequences of max_length
                for i in range(0, len(token_ids) - max_length, stride):
                    input_chunk = token_ids[i : i + max_length]
                    target_chunk = token_ids[i + 1 : i + max_length + 1]
                    if len(input_chunk) != max_length or len(target_chunk) != max_length:
                        logger.info(f"Chunk length mismatch: {len(input_chunk)} vs {max_length}")
                        logger.info(f"Inout chunk length: {len(input_chunk)}, Target chunk length: {len(target_chunk)}")

                        raise ValueError("Chunk length does not match max_length")
                    self.input_ids.append(torch.tensor(input_chunk))
                    self.target_ids.append(torch.tensor(target_chunk))
            logger.info(f"Successfully loaded and processed {len(docs_list)} documents from {file_path}")
        elif file_type == "txt":
            # Tokenize the entire text
            token_ids = tokenizer.encode(text_data, add_special_tokens=False)

            # Use a sliding window to chunk the book into overlapping sequences of max_length
            for i in range(0, len(token_ids) - max_length, stride):
                input_chunk = token_ids[i : i + max_length]
                target_chunk = token_ids[i + 1 : i + max_length + 1]
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))
            logger.info(f"Successfully loaded and processed text data from {file_path}")
        else:
            raise ValueError(f"Unsupported file_type: {file_type}")

        logger.info(f"Generated {len(self.input_ids)} sequences")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_speakleash_dataloader(
    speakleash_dataset_name="wolne_lektury_corpus",
    tokenizer=tiktoken.get_encoding("gpt2"),
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    split="train",
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
        train_ratio: Ratio of data for training (default: 0.8)
        val_ratio: Ratio of data for validation (default: 0.1)

    Returns:
        DataLoader for the specified split
    """

    dataset = SpeakleashDataLoader(
        tokenizer,
        max_length,
        stride,
        split=split,
        speakleash_dataset_name=speakleash_dataset_name,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader


def get_speaklesh_dataloader(
    split="train",
    batch_size=4,
    max_length=256,
    stride=128,
    train_ratio=0.8,
    val_ratio=0.1,
    num_workers=0,
    speakleash_dataset_name="wolne_lektury_corpus",
):
    """
    Get a specific dataloader (train, val, or test).

    Args:
        split: Which split to get ('train', 'val', or 'test')
        batch_size: Batch size for the dataloader
        max_length: Maximum sequence length
        stride: Stride for sliding window
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

    return create_speakleash_dataloader(
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        split=split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        speakleash_dataset_name=speakleash_dataset_name,
    )


def create_all_speaklesh_dataloaders(
    batch_size=4,
    max_length=256,
    stride=128,
    train_ratio=0.8,
    val_ratio=0.1,
    num_workers=0,
    speakleash_dataset_name="wolne_lektury_corpus",
):
    """
    Create train, validation, and test dataloaders (if you really need all at once).

    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader)
    """
    train_dataloader = get_speaklesh_dataloader(
        "train", batch_size, max_length, stride, train_ratio, val_ratio, num_workers, speakleash_dataset_name
    )
    val_dataloader = get_speaklesh_dataloader(
        "val", batch_size, max_length, stride, train_ratio, val_ratio, num_workers, speakleash_dataset_name
    )
    test_dataloader = get_speaklesh_dataloader(
        "test", batch_size, max_length, stride, train_ratio, val_ratio, num_workers, speakleash_dataset_name
    )

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


def create_dataloader_v1(
    txt,
    tokenizer=tiktoken.get_encoding("gpt2"),
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


if __name__ == "__main__":
    file_path = "the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    # loader = create_dataloader_v1(text_data)
    tokenizer = tiktoken.get_encoding("gpt2")
    loader = create_speakleash_dataloader(split="test", batch_size=2)
    iter = 0
    for input_ids, target_ids in loader:
        print(f"Batch {iter}:")
        print("Input IDs:", token_ids_to_text(input_ids[0][:10], tokenizer))
        print("Target IDs:", token_ids_to_text(target_ids[0][:10], tokenizer))
        # print("Input IDs shape:", input_ids[0][:10])
        # print("Target IDs shape:", target_ids[0][:10])
        iter += 1
        if iter >= 5:
            break
