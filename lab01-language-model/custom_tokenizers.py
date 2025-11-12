import json
import os
import re
from collections import Counter
from typing import List, Literal

import sentencepiece as spm

from logger import get_configured_logger

# from utils import load_split_data

logger = get_configured_logger(__name__)


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


class SimpleTokenizerBuilder:
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        # special tokens mapping to match LlamaTokenizerFast
        #     {'bos_token': '<s>',
        #   'eos_token': '<|im_end|>',
        #   'unk_token': '<unk>',
        #   'pad_token': '</s>',
        #   'additional_special_tokens': ['<|im_start|>',
        #    '<|im_end|>',
        #    '<tool_call>',
        #    '</tool_call>']},
        # convention:
        # "unk_token" -> <unk> id = 0
        # "bos_token" -> <s>' id = 1
        # "pad_token" -> </s> id = 2
        # "eos_token" -> <|im_end|> id = 4
        self.unk_token = "<unk>"
        self.unk_token_id = 0
        self.bos_token = "<s>"
        self.bos_token_id = 1
        self.pad_token = "</s>"
        self.pad_token_id = 2
        self.eos_token = "<|im_end|>"
        self.eos_token_id = 4

    def tokenize_text(self, text: str) -> List[str]:
        """
        Regex based tokenizer splitting on whitespace and punctuation
        """
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        tokens = [item.strip() for item in tokens if item.strip()]
        return tokens

    def build_vocab(self, texts: List[str] | str, save_path: str = "vocab.json"):
        """
        Build vocabulary of tokens from a list of texts

        Args:
            texts: List of texts to analyze
            save_path: Path to save the JSON vocabulary
        """
        logger.info("Tokenizing texts...")
        all_tokens = []

        # handling different type of input data: string or list of strings
        if isinstance(texts, str):
            all_tokens.extend(self.tokenize_text(texts))
        else:
            for text in texts:
                all_tokens.extend(self.tokenize_text(text))

        logger.info(f"Found {len(all_tokens)} tokens (with duplicates)")

        # Count frequency
        token_counts = Counter(all_tokens)
        logger.info(f"Unique tokens: {len(token_counts)}")

        # Reserve space for special tokens
        reserved_tokens = [self.unk_token, self.bos_token, self.pad_token, self.eos_token]
        num_reserved = len(reserved_tokens)

        # Select top (vocab_size - num_reserved) most common tokens
        most_common = token_counts.most_common(self.vocab_size - num_reserved)

        # Build vocabulary: first special tokens, then most common
        token_to_id = {}

        # Adding special tokens
        for token, idx in zip(reserved_tokens, [self.unk_token_id, self.bos_token_id, self.pad_token_id, self.eos_token_id]):
            token_to_id[token] = idx

        # manually handle situation that, id 4 is reserved for eos_token
        # and manually assign new token to id 3
        next_token, _ = most_common.pop(0)
        token_to_id[next_token] = 3
        # Add most common tokens
        for idx, (token, _) in enumerate(most_common, start=num_reserved + 1):
            token_to_id[token] = idx

        logger.info("\nVocabulary created:")
        logger.info(f"  - Size: {len(token_to_id)}")
        logger.info(f"  - Special tokens: {reserved_tokens}")
        logger.info(f"  - Most common tokens: {most_common[:10]}")
        logger.info(f"  - Rarest tokens: {most_common[-10:]}")

        # Save to JSON with metadata
        vocab_data = {
            "special_tokens": {
                "bos_token": self.bos_token,
                "unk_token": self.unk_token,
                "eos_token": self.eos_token,
                "pad_token": self.pad_token,
                "bos_token_id": self.bos_token_id,
                "unk_token_id": self.unk_token_id,
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.pad_token_id,
            },
            "vocab": token_to_id,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

        print(f"\nVocabulary saved to: {save_path}")

        return token_to_id


class SimpleTokenizer:
    def __init__(self, vocab_file_path: str):
        self.token_to_id, self.id_to_token = self.load_vocab(vocab_file_path)
        self.vocab_size = len(self.token_to_id)
        self.name = "SimpleTokenizer"

    def load_vocab(self, vocab_file_path: str):
        """
        Load vocabulary from JSON file along with special tokens
        """
        with open(vocab_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # loading special tokens and ids
        self.special_tokens = data["special_tokens"]
        self.bos_token = self.special_tokens["bos_token"]
        self.bos_token_id = self.special_tokens["bos_token_id"]
        self.unk_token = self.special_tokens["unk_token"]
        self.unk_token_id = self.special_tokens["unk_token_id"]
        self.eos_token = self.special_tokens["eos_token"]
        self.eos_token_id = self.special_tokens["eos_token_id"]
        self.pad_token = self.special_tokens["pad_token"]
        self.pad_token_id = self.special_tokens["pad_token_id"]

        # loading vocab
        token_to_id = data["vocab"]

        # Create reverse mapping
        id_to_token = {int(idx): token for token, idx in token_to_id.items()}
        return token_to_id, id_to_token

    def tokenize_text(self, text: str) -> List[str]:
        """
        Regex based tokenizer splitting on whitespace and punctuation
        """
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        tokens = [item.strip() for item in tokens if item.strip()]
        return tokens

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Convert text to a list of token IDs
        Unknown tokens are replaced with <unk>
        """
        tokens = self.tokenize_text(text)
        token_ids = [self.token_to_id.get(token, self.unk_token_id) for token in tokens]
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Convert list of token IDs back to text
        """
        tokens_to_text = [self.id_to_token.get(idx, self.unk_token) for idx in token_ids]
        text = " ".join(tokens_to_text)
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


def train_sentencepiece_tokenizer(input_file: str, model_prefix: str, vocab_size: int, model_type: Literal["unigram", "bpe"]):
    """
    Train SentencePiece tokenizer on the provided text file.

    Args:
        input_file: Path to text file with training data
        model_prefix: Prefix for output model file name
        vocab_size: Size of token vocabulary
    """
    # special tokens mapping to match LlamaTokenizerFast
    #     {'bos_token': '<s>',
    #   'eos_token': '<|im_end|>',
    #   'unk_token': '<unk>',
    #   'pad_token': '</s>',
    #   'additional_special_tokens': ['<|im_start|>',
    #    '<|im_end|>',
    #    '<tool_call>',
    #    '</tool_call>']},
    # convention:
    # "unk_token" -> <unk> id = 0
    # "bos_token" -> <s>' id = 1
    # "pad_token" -> </s> id = 2
    # "eos_token" -> <|im_end|> id = 4

    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        pad_id=2,
        unk_id=0,
        bos_id=1,
        eos_id=4,
        pad_piece="</s>",
        unk_piece="<unk>",
        bos_piece="<s>'",
        eos_piece="<|im_end|>",
        unk_surface="<unk>",
    )
    logger.info(f"Tokenizer SentencePiece was trained and saved as {model_prefix}.model and {model_prefix}.vocab")


class SentencePieceTokenizer:
    def __init__(self, model_file_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_file_path)
        self.unk_token_id = self.sp.unk_id()
        self.eos_token_id = self.sp.eos_id()
        self.bos_token_id = self.sp.bos_id()
        self.pad_token_id = self.sp.pad_id()
        self.name = "SentencePieceTokenizer"
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        token_ids = self.sp.encode_as_ids(text)
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        text = self.sp.decode_ids(token_ids)
        return text


if __name__ == "__main__":
    # training custom tokenizer
    texts = load_split_data(data_dir="train_datasets/plwiki", split="train", file_type="txt")
    builder = SimpleTokenizerBuilder(vocab_size=32000)
    vocab = builder.build_vocab(texts, save_path="tokenizers/custom_tokenizer_vocab.json")

    tokenizer = SimpleTokenizer(vocab_file_path="tokenizers/custom_tokenizer_vocab.json")
    print(tokenizer.name)
    print(f"Vocab size: {tokenizer.vocab_size}")
    sample_text = "To jest testowy tekst."
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    print(f"Sample text: {sample_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print("testing special tokens")
    print(tokenizer.decode([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 29999, 29998]))
    # # training sentencepiece tokenizer
    train_sentencepiece_tokenizer(
        input_file="train_datasets/plwiki/train.txt",
        model_prefix="plwiki_unigram",
        vocab_size=32000,
        model_type="unigram",
    )

    # # testing tokenizers
    # custom_tokenizer = SimpleTokenizer(vocab_file_path="vocab.json")
    # sp_tokenizer_bpe = SentecePieceTokenizer(model_file_path="unigram_bpe.model")
    # sp_tokenizer_bpe = spm.SentencePieceProcessor()
    # sp_tokenizer_bpe.load("plwiki_unigram.model")

    sp_tokenizer_unigram = SentencePieceTokenizer(model_file_path="plwiki_unigram.model")

    custom_text = "To jest przykładowy tekst do tokenizacji dupadupadupajas"

    # custom_encoded = custom_tokenizer.encode(custom_text)
    # custom_decoded = custom_tokenizer.decode(custom_encoded)

    # sp_encoded_bpe = sp_tokenizer_bpe.encode(custom_text)
    # sp_decoded_bpe = sp_tokenizer_bpe.decode(sp_encoded_bpe)

    sp_encoded_unigram = sp_tokenizer_unigram.encode(custom_text)
    sp_decoded_unigram = sp_tokenizer_unigram.decode(sp_encoded_unigram)

    # print("Custom Tokenizer:")
    # print(f"Text: {custom_text}")
    # print("Encoded:", custom_encoded)
    # print("Decoded:", custom_decoded)

    # print("\nSentencePiece Tokenizer (BPE):")
    # print(f"Text: {custom_text}")
    # print("Encoded:", sp_encoded_bpe)
    # print("Decoded:", sp_decoded_bpe)

    # print("\nSentencePiece Tokenizer (Unigram):")
    print(f"Text: {custom_text}")
    print("Encoded:", sp_encoded_unigram)
    print("Decoded:", sp_decoded_unigram)
    print(sp_tokenizer_unigram.decode([3]))
    print(sp_tokenizer_unigram.encode("異體字字"))
    print(sp_tokenizer_unigram.unk_token_id)
    print(sp_tokenizer_unigram.vocab_size)
    encoded = sp_tokenizer_unigram.encode("To jest przykładowy tekst do tokenizacji dupadupadupajas")
    # decoded = sp_tokenizer_unigram.decode(encoded)
    # print(f"Encoded: {encoded}")
    # print(f"Decoded: {decoded}")
    # enc = sp_tokenizer_bpe.encode(". 日本語の")
    # dec = sp_tokenizer_bpe.decode(enc)
    # print(f"BPE Encoded: {enc}")
    # print(f"BPE Decoded: {dec}")
    # enc = sp_tokenizer_bpe.encode("[EOS]")
    # dec = sp_tokenizer_bpe.decode(enc)

    # for i in range(5):
    #     print(f"ID {i} -> Token: {sp_tokenizer_bpe.id_to_piece(i)}")

    # for i in range(5):
    #     token = sp_tokenizer_bpe.id_to_piece(i)
    #     print(f"Token '{token}' -> ID: {sp_tokenizer_bpe.piece_to_id(token)}")

    # print(sp_tokenizer_bpe.piece_to_id(" "))
    # print(sp_tokenizer_bpe.encode_as_pieces("To jest przykładowy tekst do tokenizacji dupadupadupajas"))
