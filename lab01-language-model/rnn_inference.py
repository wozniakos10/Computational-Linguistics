import json
import time

import torch
from transformers import AutoTokenizer
from transformer_based_llm import generate_text_simple
from utils import get_model, get_model_config, text_to_token_ids, token_ids_to_text


def test_rnn_inference_time():
    tokenizer = AutoTokenizer.from_pretrained("speakleash/Bielik-4.5B-v3")
    vocab_size = tokenizer.n_vocab if hasattr(tokenizer, "n_vocab") else tokenizer.vocab_size
    RNN_MODEL_CONFIG = get_model_config("rnn", vocab_size=vocab_size)
    model = get_model("rnn", RNN_MODEL_CONFIG)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print("Using device:", device)
    model.load_state_dict(
        torch.load("models/rnn/best_model_checkpoint.pth", map_location=torch.device(device), weights_only=True)
    )
    model.to(device).eval()
    prompt = "Mój"
    encoded = text_to_token_ids(prompt, tokenizer).to(device)
    start = time.time()


    generated_ids = generate_text_simple(
        model,
        encoded,
        RNN_MODEL_CONFIG["max_new_tokens"],
        RNN_MODEL_CONFIG["context_length"],
        use_sampling=True,
        temperature=3,
    )
    end = time.time()
    print(encoded.shape)
    print(generated_ids.shape)
    generated_tokens = generated_ids.shape[1] - encoded.shape[1]
    print(f"Generated {generated_tokens} tokens in {end - start} seconds (1 prompt length: {encoded.shape[1]})")
    print(f"Tokens per second: {generated_tokens / (end - start)} (1 prompt length: {encoded.shape[1]})")


def test_rnn_inference_quality():
    tokenizer = AutoTokenizer.from_pretrained("speakleash/Bielik-4.5B-v3")
    vocab_size = tokenizer.n_vocab if hasattr(tokenizer, "n_vocab") else tokenizer.vocab_size
    RNN_MODEL_CONFIG = get_model_config("rnn", vocab_size=vocab_size)
    model = get_model("rnn", RNN_MODEL_CONFIG)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print("Using device:", device)
    model.load_state_dict(
        torch.load("models/rnn/best_model_checkpoint.pth", map_location=torch.device(device), weights_only=True)
    )
    model.to(device).eval()
    prompts = [
        "",
        "Polska to piękny kraj.",
        "W Polsce znajduje się wiele zabytków.",
        "Wawel to zamek królewski w Krakowie.",
    ]
    data = {}
    temperatures = [0.5, 1.0, 1.5]
    for temp in temperatures:
        data[temp] = {}
        data[temp]["prompts"] = []
        data[temp]["generations"] = []
        for p in prompts:
            print(f"Generating for temperature {temp}, prompt: '{p}'")
            data[temp]["prompts"].append(p)
            encoded = text_to_token_ids(p, tokenizer).to(device)
            from transformer_based_llm import generate_text_simple  # lub zaimplementuj generate_text_simple dla RNN

            generated_ids = generate_text_simple(
                model,
                encoded,
                RNN_MODEL_CONFIG["max_new_tokens"],
                RNN_MODEL_CONFIG["context_length"],
                use_sampling=True,
                temperature=temp,
            )
            decoded_text = token_ids_to_text(generated_ids, tokenizer)
            data[temp]["generations"].append(decoded_text)
            print("\n\n")

    with open("rnn_inference_results.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    test_rnn_inference_time()
    # test_rnn_inference_quality()
