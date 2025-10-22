from pydantic import BaseModel
from typing import Literal


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


class DataLoaderConfig(BaseModel):
    max_docs: int = 100
    use_speaklesh: bool = True
    speaklesh_dataset_name: str = "wolne_lektury_corpus"


if __name__ == "__main__":
    import tiktoken

    tokenizer = tiktoken.get_encoding("o200k_base")
    print(tokenizer)
    print("Vocab size:", tokenizer.n_vocab)
