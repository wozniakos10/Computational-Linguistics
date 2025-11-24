import time

import torch
from torch.utils.data import DataLoader

from dataset import create_huggingface_dataloader
from utils import get_classifier_model, get_classifier_model_config, get_tokenizer


@torch.no_grad()
def inference_from_scratch_trained_model(inference_dataloader: DataLoader, tokenizer, device):
    model_config = get_classifier_model_config("custom_gpt2", tokenizer, num_classes=3)
    model = get_classifier_model(model_type="custom_gpt2", model_config=model_config)
    model_path = "models/gpt2_0_44_f1_macro_classifier/best_model_checkpoint.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    model = model.to(device)
    model.eval()

    for batch in inference_dataloader:
        input_ids = batch["input_ids"].to(device)
        print(f"Input_ids shape: {input_ids.shape}")
        attention_mask = batch["attention_mask"].to(device)
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
        # just one minibatch = 100 examples
        break
    return None


@torch.no_grad()
def inference_fine_tuned_model(inference_dataloader: DataLoader, tokenizer, device):
    model_config = get_classifier_model_config("hugging_face", tokenizer, num_classes=3)
    model = get_classifier_model(model_type="hugging_face", model_config=model_config)
    model_path = "models/bielik_0_62_f1_macro_classfier/best_model_checkpoint.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    model = model.to(device)
    model.eval()

    for batch in inference_dataloader:
        input_ids = batch["input_ids"].to(device)
        print(f"Input_ids shape: {input_ids.shape}")
        attention_mask = batch["attention_mask"].to(device)
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
        # just one minibatch = 100 examples
        break
    return None


def compare_inferences_times():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    batch_size = 100
    print(f"Using device: {device}")
    tokenizer = get_tokenizer("speakleash/Bielik-1.5B-v3", tokenizer_type="transformers")
    inference_dataloder = create_huggingface_dataloader(
        dataset_name="jziebura/polish_youth_slang_classification", split="train", tokenizer=tokenizer, max_length=256, batch_size=batch_size
    )

    start = time.time()
    inference_from_scratch_trained_model(inference_dataloader=inference_dataloder, tokenizer=tokenizer, device=device)
    stop = time.time()
    print(f"Inference for {batch_size} examples for trained from scratch model took: {stop - start:.3f}\n\n")

    start = time.time()
    inference_fine_tuned_model(inference_dataloader=inference_dataloder, tokenizer=tokenizer, device=device)
    stop = time.time()
    print(f"Inference for {batch_size} examples for fine-tuned model took: {stop - start:.3f}")


if __name__ == "__main__":
    compare_inferences_times()
