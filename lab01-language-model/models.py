from typing import Literal

from pydantic import BaseModel


class CustomDecoderClassifierConfig(BaseModel):
    vocab_size: int
    context_length: int
    num_classes: int
    unfreeze_last_n_layers: int


class TransformerModelClassifierConfig(BaseModel):
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate: float
    num_classes: int
    qkv_bias: bool


class TransformerModelConfig(BaseModel):
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate: float
    qkv_bias: bool
    max_new_tokens: int


class RnnModelConfig(BaseModel):
    vocab_size: int
    emb_dim: int
    rnn_hidden_dim: int
    num_layers: int
    drop_rate: float
    context_length: int
    max_new_tokens: int


class ModelTrainingConfig(BaseModel):
    learning_rate: float
    num_epochs: int
    batch_size: int
    weight_decay: float
    optimizer: Literal["adamw", "sgd"] = "adamw"
    eval_freq: int = 20
    eval_iter: int = 5
    max_training_minutes: int = 60
    gradient_clip: float = 1


class DataLoaderConfig(BaseModel):
    use_speaklesh: bool | None = None
    speaklesh_dataset_name: str | None = None
    use_hugging_face: bool | None = None
    hugging_face_dataset_name: str | None = None


if __name__ == "__main__":
    import tiktoken

    tokenizer = tiktoken.get_encoding("o200k_base")
    print(tokenizer)
    print("Vocab size:", tokenizer.n_vocab)
