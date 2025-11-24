# Playing with and adjusting for my use-case: https://github.com/rasbt/LLMs-from-scratch

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel

from logger import get_configured_logger

logger = get_configured_logger(__name__)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)

        # To maskowanie de facto tworzy nam dekoder - generujac przyszly tokeny nie widzimy przyszlosci i
        # zakrywamy jes maska w postaci macierzy trojkatnej gornej
        #
        # [token, -inf, -inf]
        # [token, token, -inf]
        # [token, token, token]
        #
        # -inf daje sie dlatego, ze potem maska trafia do softmax i mamy exp(-inf) = 0, czyli nie bierzemy pod uwage
        # tych wartosci przy sumowaniu.
        # Jest to self attention, bo atencje wykonujemy na tej samej sewkencji - dostajemy wtedy informacje
        # jak sekwencja wplywa sama na siebie. Jezeli bylby dekoder, to nie mamy maski i moglibysmy patrzec na cala
        # sewkencje np. do okreslania sentymentu. Cross attention to interakcja miedzy roznymmi sekwencjami
        # np w attentsion is all you need mamy encoder-decoder i cross attention w dekoderze ale to
        # dlatego ze tam bylo zadanie tlumaczenia jezyka na inny i mozna to tak zastosowac.

        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # from scratch
        # self.att = MultiHeadAttention(
        #     d_in=cfg["emb_dim"],
        #     d_out=cfg["emb_dim"],
        #     context_length=cfg["context_length"],
        #     num_heads=cfg["n_heads"],
        #     dropout=cfg["drop_rate"],
        #     qkv_bias=cfg["qkv_bias"])

        # pytorch module
        self.att = nn.MultiheadAttention(
            embed_dim=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            bias=cfg["qkv_bias"],
            batch_first=True,
        )
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block (redisual connection)
        shortcut = x
        x = self.norm1(x)

        # # Create causal mask for self-attention
        seq_len = x.size(1)
        # THE TRICKY PART OF ADDING MASK TO PYTORCH MODULES:
        # https://sanjayasubedi.com.np/deeplearning/masking-in-attention/
        # In Pytorch scaled_dot_product_attention function when a boolean mask is passed to attn_mask parameter,
        # a value of True indicates that the element should take part in attention. However in MultiHeadAttention Layer,
        # TransformerEncoderLayer and TransformerDecoderLayer for a binary mask, a True value indicates that the corresponding key
        #     value will be ignored for the purpose of attention. Not sure why they implemented it differently, but I will consider
        #     True value to be ignored during attention calculation.
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(x.device)

        # PyTorch MultiheadAttention expects (query, key, value) and returns (output, attention_weights)
        # z racji ze to self attetion, zamiast q,k,v podajemy to samo x
        x, _ = self.att(x, x, x, attn_mask=causal_mask, need_weights=False)

        # for sratch implementation
        # x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # self.trf_blocks = nn.Transformer(
        #     d_model=cfg["emb_dim"],
        #     nhead=cfg["n_heads"],
        #     num_encoder_layers=0,
        #     num_decoder_layers=cfg["n_layers"],
        #     dim_feedforward=4 * cfg["emb_dim"],
        #     dropout=cfg["drop_rate"],
        #     activation="gelu"
        # )

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class GPTModelSequenceClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = nn.LayerNorm(cfg["emb_dim"])

        self.classifier_head = nn.Sequential(
            nn.Linear(cfg["emb_dim"], cfg["emb_dim"]),
            nn.GELU(),
            nn.Dropout(cfg["drop_rate"]),
            nn.Linear(cfg["emb_dim"], cfg["num_classes"]),
        )

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        tok_embeds = self.tok_emb(input_ids)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=input_ids.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        # Find last non padding token representation for each sequence in the batch
        if attention_mask is not None:
            # Sum of attention_mask gives the number of real tokens
            # Subtract 1 because we index from 0
            sequence_lengths = attention_mask.sum(dim=1) - 1

            # Get the representation of the last real tokens
            batch_indices = torch.arange(batch_size, device=input_ids.device)
            last_token_hidden = x[batch_indices, sequence_lengths]
        else:
            # Fallback - use the last token (as before)
            last_token_hidden = x[:, -1, :]

        classification_logits = self.classifier_head(last_token_hidden)

        return classification_logits


class CustomDecoderClassifier(nn.Module):
    def __init__(self, checkpoint, num_labels, unfreeze_last_n_layers=3):
        super().__init__()

        # 1. Load backbone
        self.backbone = AutoModel.from_pretrained(checkpoint)

        # --- (FREEZING) ---

        # Freeze all weights in the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Find transformer layers in the backbone
        if hasattr(self.backbone, "layers"):
            layers_to_train = self.backbone.layers
        elif hasattr(self.backbone, "h"):  # Deal with gpt2
            layers_to_train = self.backbone.h
        else:
            raise AttributeError("Can't find transformer layers in the backbone model.")

        # Step C: Unfreeze the last X layers
        # Take a slice [-5:] and set requires_grad = True
        if unfreeze_last_n_layers > 0:
            for layer in layers_to_train[-unfreeze_last_n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

            # Unfreezing also the final layer norm if present
            if hasattr(self.backbone, "norm"):
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
            elif hasattr(self.backbone, "ln_f"):  # GPT-2
                for param in self.backbone.ln_f.parameters():
                    param.requires_grad = True

        logger.info(f"Successfully froze backbone and unfroze last {unfreeze_last_n_layers} layers.")
        # ------------------------------------

        hidden_size = self.backbone.config.hidden_size

        # 2. Custom classifier head
        self.classifier = nn.Sequential(nn.Linear(hidden_size, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.4), nn.Linear(512, num_labels))

    def forward(self, input_ids, attention_mask=None):
        # Flow through backbone
        # output.last_hidden_state has shape [batch_size, seq_len, hidden_size]
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # 3. Extract the correct vector (last token representation)
        if attention_mask is not None:
            # Calculate indices of the last tokens (non-padding)
            # -1 because indices are zero-based
            last_token_indices = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.shape[0]

            # Get the representation of the last real tokens
            batch_indices = torch.arange(batch_size, device=input_ids.device)
            # Select the appropriate vectors for each element in the batch
            # Fancy indexing: [0..batch, last_indices, :]
            embedding = last_hidden_state[batch_indices, last_token_indices]
        else:
            # If there is no mask (no padding), simply take the last element
            embedding = last_hidden_state[:, -1, :]

        # 4. Flow through the head
        logits = self.classifier(embedding)

        return logits


def temperature_scaled_softmax(logits, temperature=1.0):
    logits = logits / temperature
    return F.softmax(logits, dim=-1)


def generate_text_simple(model, idx, max_new_tokens, context_size, temperature=1.0, use_sampling=False):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        # print(f"idx_cond shape: {idx_cond.shape}")
        # print(f"idx shape: {idx.shape}")
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        # print(f"idx shape: {idx_cond.shape}, logits shape: {logits.shape}")
        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # apply softmax to get probabilities
        # probs = F.softmax(logits, dim=-1) # (B, C)

        if use_sampling:
            # Custom implementation with temperature scaling
            probs = temperature_scaled_softmax(logits, temperature)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

        else:
            # Get the idx of the vocab entry with the highest logits value
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def main():
    from torchinfo import summary

    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 256,  # Context length
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False,  # Query-Key-Value bias
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # disable dropout
    # Print model summary - GPT model expects (batch_size, sequence_length) input of token indices
    # summary(model, input_size=(1, 13))

    vocab_size = 5000  # ustaw zgodnie z modelem
    seq_len = 256  # długość sekwencji (tak jak wcześniej)

    # zamiast torch.randn → użyj losowych indeksów całkowitych
    dummy_input = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)

    # teraz użyj torchinfo.summary zamiast torchsummary.summary
    summary(model, input_data=dummy_input)

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("o200k_base")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50 * '='}\n{22 * ' '}IN\n{50 * '='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(model=model, idx=encoded_tensor, max_new_tokens=10, context_size=GPT_CONFIG_124M["context_length"])
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50 * '='}\n{22 * ' '}OUT\n{50 * '='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)


if __name__ == "__main__":
    main()
