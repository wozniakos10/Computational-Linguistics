from transformer_based_llm import generate_text_simple
import torch
from transformers import AutoTokenizer
from utils import text_to_token_ids, token_ids_to_text, get_model_config, get_model
import time
import json


def test_transformer_inference_time():
    tokenizer = AutoTokenizer.from_pretrained("speakleash/Bielik-4.5B-v3")
    vocab_size = tokenizer.n_vocab if hasattr(tokenizer, "n_vocab") else tokenizer.vocab_size
    TRANSFORMER_MODEL_CONFIG = get_model_config("transformer", vocab_size=vocab_size)
    model = get_model("transformer", TRANSFORMER_MODEL_CONFIG)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print("Using device:", device)
    # mapping to load model saved with CUDA on CPU or MPS
    model.load_state_dict(torch.load("models/transformer/best_model_checkpoint.pth", map_location=torch.device(device), weights_only=True))
    model.to(device).eval()
    prompt = "Mój"
    encoded = text_to_token_ids(prompt, tokenizer).to(device)
    start = time.time()
    generated_ids = generate_text_simple(
        model,
        encoded,
        TRANSFORMER_MODEL_CONFIG["max_new_tokens"],
        TRANSFORMER_MODEL_CONFIG["context_length"],
        use_sampling=True,
        temperature=3,
    )
    end = time.time()
    print(encoded.shape)
    print(generated_ids.shape)
    generated_tokens = generated_ids.shape[1] - encoded.shape[1]
    print(f"Generated {generated_tokens} tokens in {end - start} seconds (1 prompt length: {encoded.shape[1]})")
    print(f"Tokens per second: {generated_tokens / (end - start)} (1 prompt length: {encoded.shape[1]})")


def test_transformer_inference_quality():
    tokenizer = AutoTokenizer.from_pretrained("speakleash/Bielik-4.5B-v3")
    vocab_size = tokenizer.n_vocab if hasattr(tokenizer, "n_vocab") else tokenizer.vocab_size
    TRANSFORMER_MODEL_CONFIG = get_model_config("transformer", vocab_size=vocab_size)
    model = get_model("transformer", TRANSFORMER_MODEL_CONFIG)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print("Using device:", device)
    # mapping to load model saved with CUDA on CPU or MPS
    model.load_state_dict(torch.load("models/transformer/best_model_checkpoint.pth", map_location=torch.device(device), weights_only=True))
    model.to(device).eval()
    prompt = "Mój"
    encoded = text_to_token_ids(prompt, tokenizer).to(device)
    generated_ids = generate_text_simple(
        model,
        encoded,
        TRANSFORMER_MODEL_CONFIG["max_new_tokens"],
        TRANSFORMER_MODEL_CONFIG["context_length"],
        use_sampling=True,
        temperature=3,
    )
    # prompts = [
    #     "",
    #     "Polska to piękny kraj.",
    #     "W Polsce znajduje się wiele zabytków.",
    #     "Wawel to zamek królewski w Krakowie.",
    #
    prompts = [
        "Polska to piękny kraj polozony w eruopie srodkowej nad rzeka wisla. Nasza historia sięga 966 roku kiedy mieszko 1 przyjął chrzest."
    ]
    data = {}
    temperatures = [0.7, 1.0, 1.3]
    for temp in temperatures:
        data[temp] = {}
        data[temp]["prompts"] = []
        data[temp]["generations"] = []
        for p in prompts:
            print(f"Generating for temperature {temp}, prompt: '{p}'")
            data[temp]["prompts"].append(p)
            encoded = text_to_token_ids(p, tokenizer).to(device)
            generated_ids = generate_text_simple(
                model,
                encoded,
                TRANSFORMER_MODEL_CONFIG["max_new_tokens"],
                TRANSFORMER_MODEL_CONFIG["context_length"],
                use_sampling=True,
                temperature=temp,
            )
            decoded_text = token_ids_to_text(generated_ids, tokenizer)
            data[temp]["generations"].append(decoded_text)
            print("\n\n")
            print(data)

    with open("transformer_inference_results.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # test_transformer_inference_time()
    test_transformer_inference_quality()
