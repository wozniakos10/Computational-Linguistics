from transformer_based_llm import GPTModel, generate_text_simple
import torch
from models import TransformerModelConfig, ModelTrainingConfig, DataLoaderConfig
from transformers import AutoTokenizer
from utils import text_to_token_ids, token_ids_to_text


if __name__ == "__main__":
    print("test")
    # tokenizer = tiktoken.get_encoding("o200k_base")
    tokenizer = AutoTokenizer.from_pretrained("flax-community/papuGaPT2")
    vocab_size = tokenizer.n_vocab if hasattr(tokenizer, "n_vocab") else tokenizer.vocab_size
    TRANSFORMER_MODEL_CONFIG = TransformerModelConfig(
        **{
            "vocab_size": vocab_size,  # Vocabulary size
            "context_length": 256,  # Shortened context length (orig: 1024)
            "emb_dim": 768,  # Embedding dimension
            "n_heads": 12,  # Number of attention heads
            "n_layers": 12,  # Number of layers
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
    model.load_state_dict(torch.load("models/model_id_28b6bf7cd81741439a62aa38b901fb72.pth", weights_only=True))

    prompt = "Mój ulubiony sport to piłka nożna, bardzo lubię oglądać mecze na żywo i kibicować"
    encoded = text_to_token_ids(prompt, tokenizer).to(device)
    # generated_ids = generate_text_simple(
    #     model,
    #     encoded,
    #     TRANSFORMER_MODEL_CONFIG["max_new_tokens"],
    #     TRANSFORMER_MODEL_CONFIG["context_length"],
    #     use_sampling=True,
    #     temperature=1,
    # )
    # decoded_text = token_ids_to_text(generated_ids, tokenizer)
    # print("Początek:", prompt)
    # print("Wygenerowany tekst with sampling:", decoded_text)
    # print("", end="\n\n")

    generated_ids = generate_text_simple(
        model,
        encoded,
        TRANSFORMER_MODEL_CONFIG["max_new_tokens"],
        TRANSFORMER_MODEL_CONFIG["context_length"],
        use_sampling=True,
        temperature=3,
    )
    decoded_text = token_ids_to_text(generated_ids, tokenizer)
    print("Początek:", prompt)
    print("Wygenerowany tekst without sampling:", decoded_text)
