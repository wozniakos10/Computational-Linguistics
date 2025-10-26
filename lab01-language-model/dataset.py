import os

import tiktoken
import torch
from speakleash import Speakleash
from speakleash.dataset import SpeakleashDataset
from torch.utils.data import DataLoader, Dataset


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


class SpeakleashDataLoader(Dataset):
    def __init__(
        self,
        dataset: SpeakleashDataset,
        tokenizer,
        max_length,
        stride,
        max_docs,
        split="train",
        train_ratio=0.8,
        val_ratio=0.1,
        only_high_quality=True,
    ):
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
        self.dataset = dataset
        self.dataset_data = self.dataset.ext_data

        self.only_high_quality = only_high_quality
        # Calculate split indices
        self.max_docs = min(max_docs, self.dataset.manifest["stats"]["documents"])

        self.input_ids = []
        self.target_ids = []

        # Process documents for the specified split with error handling
        processed_docs = 0
        skipped_docs = 0
        current_doc_index = 0
        docs_list = []
        low_quality_count = 0
        for doc in self.dataset_data:
            # Check if we've reached our max_docs limit
            if len(docs_list) >= self.max_docs:
                break
            text, meta = doc
            quality = meta.get("quality", "")
            if self.only_high_quality:
                if quality == "HIGH":
                    docs_list.append(text)
                else:
                    low_quality_count += 1
            else:
                docs_list.append(text)

        new_max_docs = min(self.max_docs, len(docs_list))
        train_end = int(new_max_docs * train_ratio)
        val_end = int(new_max_docs * (train_ratio + val_ratio))

        if split == "train":
            doc_range = range(0, train_end)
        elif split == "val":
            doc_range = range(train_end, val_end)
        elif split == "test":
            doc_range = range(val_end, self.max_docs)
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        for doc in docs_list:
            if current_doc_index not in doc_range:
                current_doc_index += 1
                continue

            try:
                token_ids = tokenizer.encode(doc)

                # Use a sliding window to chunk the document into overlapping sequences of max_length
                for i in range(0, len(token_ids) - max_length, stride):
                    input_chunk = token_ids[i : i + max_length]
                    target_chunk = token_ids[i + 1 : i + max_length + 1]
                    self.input_ids.append(torch.tensor(input_chunk))
                    self.target_ids.append(torch.tensor(target_chunk))

                processed_docs += 1
            except Exception as e:
                print(f"Skipping corrupted document at index {current_doc_index}: {e}")
                skipped_docs += 1
            finally:
                current_doc_index += 1

        print(
            f"Successfully processed {processed_docs} documents for '{split}'"
            f"split, skipped {skipped_docs} corrupted documents\nskipped {low_quality_count} low-quality documents"
        )
        print(f"Generated {len(self.input_ids)} sequences")

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
    speaklesh_dataset = speakleash.get(speakleash_dataset_name)
    print(f"Loaded Speakleash dataset '{speakleash_dataset_name}' with {speaklesh_dataset.manifest['stats']['documents']} documents")
    dataset = SpeakleashDataLoader(
        speaklesh_dataset,
        tokenizer,
        max_length,
        stride,
        max_docs,
        split=split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader


def get_speaklesh_dataloader(
    split="train",
    batch_size=4,
    max_length=256,
    stride=128,
    max_docs=100,
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

    return create_speakleash_dataloader(
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
        speakleash_dataset_name=speakleash_dataset_name,
    )


def create_all_speaklesh_dataloaders(
    batch_size=4,
    max_length=256,
    stride=128,
    max_docs=100,
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
        "train", batch_size, max_length, stride, max_docs, train_ratio, val_ratio, num_workers, speakleash_dataset_name
    )
    val_dataloader = get_speaklesh_dataloader(
        "val", batch_size, max_length, stride, max_docs, train_ratio, val_ratio, num_workers, speakleash_dataset_name
    )
    test_dataloader = get_speaklesh_dataloader(
        "test", batch_size, max_length, stride, max_docs, train_ratio, val_ratio, num_workers, speakleash_dataset_name
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
    loader = create_speakleash_dataloader(split="test", batch_size=2, max_docs=1000)
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
