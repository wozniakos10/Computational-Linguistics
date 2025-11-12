import json
import os
from typing import Literal

import tiktoken
import torch
import torch.nn.functional as F
from speakleash import Speakleash
from transformers import AutoTokenizer

from custom_tokenizers import SentencePieceTokenizer, SimpleTokenizer
from logger import get_configured_logger
from models import RnnModelConfig, TransformerModelConfig
from rnn_based_llm import RnnBasedModel
from transformer_based_llm import GPTModel, generate_text_simple

logger = get_configured_logger("llm_train", log_file="logs/llm_train.log")


def get_model(model_type: Literal["rnn", "transformer"], model_config: dict):
    if model_type == "rnn":
        return RnnBasedModel(
            vocab_size=model_config["vocab_size"],
            emb_dim=model_config["emb_dim"],
            rnn_hidden_dim=model_config["rnn_hidden_dim"],
            num_layers=model_config["num_layers"],
            drop_rate=model_config["drop_rate"],
        )
    elif model_type == "transformer":
        return GPTModel(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_config(model_type: Literal["rnn", "transformer"], vocab_size: int = 50257):
    if model_type == "rnn":
        return RnnModelConfig(
            vocab_size=vocab_size,
            emb_dim=512,
            rnn_hidden_dim=1024,
            num_layers=10,
            drop_rate=0.3,
            context_length=256,
            max_new_tokens=256,
        ).model_dump()
    elif model_type == "transformer":
        return TransformerModelConfig(
            vocab_size=vocab_size,
            context_length=256,
            emb_dim=768,
            n_heads=12,
            n_layers=2,
            drop_rate=0.2,
            qkv_bias=False,
            max_new_tokens=256,
        ).model_dump()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_tokenizer(tokenizer_name: str, tokenizer_type: Literal["tiktoken", "transformers", "sentence_piece", "custom"]):
    if tokenizer_type == "tiktoken":
        logger.info(f"Using tiktoken tokenizer: {tokenizer_name}")
        return tiktoken.get_encoding(tokenizer_name)
    elif tokenizer_type == "transformers":
        logger.info(f"Using Hugging Face tokenizer: {tokenizer_name}")
        return AutoTokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_type == "sentence_piece":
        logger.info(f"Using SentencePiece tokenizer: {tokenizer_name}")
        return SentencePieceTokenizer(model_file_path=tokenizer_name)
    elif tokenizer_type == "custom":
        logger.info(f"Using Custom tokenizer: {tokenizer_name}")
        return SimpleTokenizer(tokenizer_name)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, add_special_tokens=False)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device, tokenizer, calculate_perplexity=False):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    if calculate_perplexity:
        # sum loss for perplexity calculation
        sum_loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten(), reduction="sum")

        pred_tokens = torch.argmax(logits, dim=-1, keepdim=True)

        # # reshape for fit function
        decoded_text = token_ids_to_text(pred_tokens.flatten(0, 1).squeeze(1), tokenizer)
        print(decoded_text)
        # normzalize with batch size
        char_amount = len(decoded_text)

        # calculating words just by splitting by space and normalize with batch size
        words_amount = len(decoded_text.split())
        mean_loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

        # This line gave same results as: torch.epx(mean_loss)
        perplexity_by_token = torch.exp(sum_loss / (input_batch.shape[0] * input_batch.shape[1]))
        perplexity_by_char = torch.exp(sum_loss / char_amount)
        perplexity_by_word = torch.exp(sum_loss / words_amount)

        return mean_loss, perplexity_by_token, perplexity_by_char, perplexity_by_word

    else:
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

        return loss


def calc_loss_loader(data_loader, model, device, tokenizer, num_batches=None):
    total_loss = 0.0
    perplexity_per_token = 0.0
    perplexity_per_char = 0.0
    perplexity_per_word = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss, per_per_token, per_per_char, per_per_word = calc_loss_batch(
                input_batch, target_batch, model, device, tokenizer, calculate_perplexity=True
            )
            total_loss += loss.item()
            perplexity_per_token += per_per_token
            perplexity_per_char += per_per_char
            perplexity_per_word += per_per_word
        else:
            break
    return (
        total_loss / num_batches,
        perplexity_per_token / num_batches,
        perplexity_per_char / num_batches,
        perplexity_per_word / num_batches,
    )


def evaluate_model(model, train_loader, val_loader, device, eval_iter, tokenizer):
    model.eval()
    with torch.no_grad():
        train_loss, train_perp_token, train_perp_char, train_perp_word = calc_loss_loader(
            train_loader, model, device, tokenizer, num_batches=eval_iter
        )
        val_loss, val_perp_token, val_perp_char, val_perp_word = calc_loss_loader(
            val_loader, model, device, tokenizer, num_batches=eval_iter
        )
    model.train()
    train_perplexity = {
        "per_token": train_perp_token,
        "per_char": train_perp_char,
        "per_word": train_perp_word,
    }
    val_perplexity = {
        "per_token": val_perp_token,
        "per_char": val_perp_char,
        "per_word": val_perp_word,
    }
    return train_loss, val_loss, train_perplexity, val_perplexity


def generate_and_print_sample(model: GPTModel, tokenizer, device, start_context, max_new_tokens=50):
    model.eval()
    if hasattr(model, "pos_emb"):
        context_size = model.pos_emb.weight.shape[0]
    else:
        # fallback for RNNs if not provided
        context_size = max_new_tokens
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=max_new_tokens, context_size=context_size, use_sampling=True
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        logger.info(f"Sample text generation with context: '{start_context}'\ngenerated text: '{decoded_text.replace('\n', ' ')}'")
    model.train()


def temperature_scaled_softmax(logits, temperature=1.0):
    logits = logits / temperature
    return F.softmax(logits, dim=-1)


def extract_speakleash_data_to_files(dataset_name: str, output_dir: str, max_docs: int, only_high_quality=True, save_as_txt=False) -> None:
    # Initialize Speakleash and get dataset
    base_dir = os.path.join(os.path.dirname(__file__))
    replicate_to = os.path.join(base_dir, "datasets")
    speakleash = Speakleash(replicate_to)
    speaklesh_dataset = speakleash.get(dataset_name)

    # Test ratio is automatically rest of data after that split
    train_ratio = 0.8
    val_ratio = 0.1

    dataset_data = speaklesh_dataset.ext_data

    # Calculate split indices
    max_docs = min(max_docs, speaklesh_dataset.manifest["stats"]["documents"])

    docs_list = []
    low_quality_count = 0
    for doc in dataset_data:
        # Check if we've reached our max_docs limit
        if len(docs_list) >= max_docs:
            break
        text, meta = doc
        quality = meta.get("quality", "")
        if only_high_quality:
            if quality == "HIGH":
                docs_list.append(text)
            else:
                low_quality_count += 1
        else:
            docs_list.append(text)

    # Calculate split indices
    new_max_docs = len(docs_list)
    train_end = int(new_max_docs * train_ratio)
    val_end = int(new_max_docs * (train_ratio + val_ratio))

    # Split the data
    train_docs = docs_list[:train_end]
    val_docs = docs_list[train_end:val_end]
    test_docs = docs_list[val_end:]

    # Create output directory if it doesn't exist
    output_dir = os.path.join(base_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Save files based on format
    if save_as_txt:
        # Save as TXT files (one document per line or separated by newlines)
        with open(os.path.join(output_dir, "train.txt"), "w", encoding="utf-8") as f:
            for doc in train_docs:
                f.write(doc + "\n")

        with open(os.path.join(output_dir, "val.txt"), "w", encoding="utf-8") as f:
            for doc in val_docs:
                f.write(doc + "\n")

        with open(os.path.join(output_dir, "test.txt"), "w", encoding="utf-8") as f:
            for doc in test_docs:
                f.write(doc + "\n")

        file_format = "TXT"
    else:
        # Save as JSONL files
        with open(os.path.join(output_dir, "train.jsonl"), "w", encoding="utf-8") as f:
            for doc in train_docs:
                f.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")

        with open(os.path.join(output_dir, "val.jsonl"), "w", encoding="utf-8") as f:
            for doc in val_docs:
                f.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")

        with open(os.path.join(output_dir, "test.jsonl"), "w", encoding="utf-8") as f:
            for doc in test_docs:
                f.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")

        file_format = "JSONL"

    logger.info(f"Files saved as: {file_format}")
    logger.info(f"Total documents: {new_max_docs}")
    logger.info(f"Train: {len(train_docs)} documents ({len(train_docs) / new_max_docs * 100:.1f}%)")
    logger.info(f"Val: {len(val_docs)} documents ({len(val_docs) / new_max_docs * 100:.1f}%)")
    logger.info(f"Test: {len(test_docs)} documents ({len(test_docs) / new_max_docs * 100:.1f}%)")
    if only_high_quality:
        logger.info(f"Low quality documents skipped: {low_quality_count}")


def load_split_data(data_dir: str, split: str = "train", file_type: str = "jsonl") -> list[str] | str:
    """
    Load train, val, or test data from JSONL or TXT files.

    Args:
        data_dir: Directory path where the split files are stored
        split: Which split to load - 'train', 'val', or 'test'
        file_type: File format - 'jsonl' or 'txt' (default: 'jsonl')

    Returns:
        List of documents (strings)
    """

    if file_type not in ["jsonl", "txt"]:
        raise ValueError(f"Unsupported file_type: {file_type}. Must be 'jsonl' or 'txt'")

    file_path = os.path.join(data_dir, f"{split}.{file_type}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_type == "jsonl":
        docs = []
        # Load JSONL format
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    doc_data = json.loads(line)
                    docs.append(doc_data["text"])
        logger.info(f"Loaded {len(docs)} documents from {split}.{file_type}")
    else:
        # Load TXT format (one document per line)
        with open(file_path, "r", encoding="utf-8") as f:
            docs = f.read()
        logger.info(f"Loaded {len(docs)} characters from {split}.{file_type}")

    return docs


if __name__ == "__main__":
    # import tiktoken

    # tokenizer = get_tokenizer("speakleash/Bielik-4.5B-v3", is_tiktoken=False)
    # device = "mps"
    # input_text = "Polski język jest bardzo trudny?"
    # input_ids = text_to_token_ids(input_text, tokenizer).to(device)
    # print(f"Input text: {input_text}")
    # print(f"Input IDs: {input_ids}")
    # # Convert input_ids to 1D list for decoding
    # decoded_text = tokenizer.decode(input_ids.squeeze().tolist(), skip_special_tokens=True)
    # print(f"Decoded text: {decoded_text}")
    # print(tokenizer.decode([1]))  # special token
    # print(tokenizer.decode([31956]))  # special token
    extract_speakleash_data_to_files(
        dataset_name="plwiki",
        output_dir="train_datasets/plwiki_test",
        max_docs=1000,
        only_high_quality=True,
        save_as_txt=True,
    )
    # data = load_split_data(data_dir="train_datasets/plwiki", split="train", file_type="txt")
    # print(data.count("\n"))
    # print(type(data))
    # print(data)
