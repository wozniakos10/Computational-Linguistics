from transformer_based_llm import GPTModel, generate_text_simple
import torch
import tiktoken
from models import TransformerModelConfig, ModelTrainingConfig, DataLoaderConfig


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


if __name__ == "__main__":
    print("test")
    tokenizer = tiktoken.get_encoding("gpt2")
    TRANSFORMER_MODEL_CONFIG = TransformerModelConfig(
        **{
            "vocab_size": tokenizer.n_vocab,  # Vocabulary size
            "context_length": 256,  # Shortened context length (orig: 1024)
            "emb_dim": 768,  # Embedding dimension
            "n_heads": 12,  # Number of attention heads
            "n_layers": 14,  # Number of layers
            "drop_rate": 0.35,  # Dropout rate
            "qkv_bias": False,  # Query-key-value bias
            "max_new_tokens": 256,
        }
    ).model_dump()

    TRAINING_SETTINGS = ModelTrainingConfig(
        **{
            "learning_rate": 45e-4,
            "num_epochs": 5,
            "batch_size": 4,
            "weight_decay": 0.5,
            "optimizer": "adamw",
        }
    ).model_dump()

    DATASET_SETTINGS = DataLoaderConfig(**{"max_docs": 100, "use_speaklesh": True}).model_dump()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    model = GPTModel(TRANSFORMER_MODEL_CONFIG).to(device)
    model.load_state_dict(torch.load("models/model_id_7ac77f258b0545cbb0443e90e584835c.pth", weights_only=True))

    prompt = "Pierogi ruskie to moje ulubione danie, ponieważ"
    encoded = text_to_token_ids(prompt, tokenizer).to(device)
    generated_ids = generate_text_simple(
        model,
        encoded,
        TRANSFORMER_MODEL_CONFIG["max_new_tokens"],
        TRANSFORMER_MODEL_CONFIG["context_length"],
        use_sampling=True,
        temperature=1,
    )
    decoded_text = token_ids_to_text(generated_ids, tokenizer)
    print("Początek:", prompt)
    print("Wygenerowany tekst with sampling:", decoded_text)
    print("", end="\n\n")

    generated_ids = generate_text_simple(
        model,
        encoded,
        TRANSFORMER_MODEL_CONFIG["max_new_tokens"],
        TRANSFORMER_MODEL_CONFIG["context_length"],
        use_sampling=False,
    )
    decoded_text = token_ids_to_text(generated_ids, tokenizer)
    print("Początek:", prompt)
    print("Wygenerowany tekst without sampling:", decoded_text)
