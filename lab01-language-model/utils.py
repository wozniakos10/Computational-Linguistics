from typing import Literal

import mlflow
import tiktoken
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

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
            emb_dim=128,
            rnn_hidden_dim=256,
            num_layers=3,
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
            n_layers=12,
            drop_rate=0.2,
            qkv_bias=False,
            max_new_tokens=256,
        ).model_dump()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_tokenizer(tokenizer_name: str, is_tiktoken: bool = False):
    if is_tiktoken:
        logger.info(f"Using tiktoken tokenizer: {tokenizer_name}")
        return tiktoken.get_encoding(tokenizer_name)
    else:
        logger.info(f"Using Hugging Face tokenizer: {tokenizer_name}")
        return AutoTokenizer.from_pretrained(tokenizer_name)


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    train_perplexity = torch.exp(torch.tensor(train_loss))
    val_perplexity = torch.exp(torch.tensor(val_loss))
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
        mlflow.log_text(decoded_text, "artifacts/generated_text.txt")
    model.train()


def temperature_scaled_softmax(logits, temperature=1.0):
    logits = logits / temperature
    return F.softmax(logits, dim=-1)


if __name__ == "__main__":
    import tiktoken

    tokenizer = get_tokenizer("speakleash/Bielik-4.5B-v3", is_tiktoken=False)
    device = "mps"
    input_text = "Polski język jest bardzo trudny?"
    input_ids = text_to_token_ids(input_text, tokenizer).to(device)
    print(f"Input text: {input_text}")
    print(f"Input IDs: {input_ids}")
    # Convert input_ids to 1D list for decoding
    decoded_text = tokenizer.decode(input_ids.squeeze().tolist(), skip_special_tokens=True)
    print(f"Decoded text: {decoded_text}")
    print(tokenizer.decode([1]))  # special token
    print(tokenizer.decode([31956]))  # special token
