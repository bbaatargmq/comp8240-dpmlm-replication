import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
import numpy as np

# DP parameters
EPSILON = 50
CLIP_MIN = -10
CLIP_MAX = 10

# Load tokenizer & model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForMaskedLM.from_pretrained("roberta-base")
model.eval()

def dp_masked_prediction(sentence, epsilon=EPSILON):
    tokens = tokenizer(sentence, return_tensors="pt")
    input_ids = tokens["input_ids"].clone()

    # Mask the first non-special token
    mask_index = (input_ids[0] != tokenizer.cls_token_id).nonzero()[0].item()
    input_ids[0][mask_index] = tokenizer.mask_token_id

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, mask_index]

    # Clip logits (DP sensitivity bounding)
    logits = torch.clamp(logits, CLIP_MIN, CLIP_MAX)

    # Temperature sampling (Exponential mechanism)
    T = (CLIP_MAX - CLIP_MIN) / epsilon
    probs = torch.softmax(logits / T, dim=0)

    # Sample DP token
    sampled_id = torch.multinomial(probs, 1).item()
    dp_token = tokenizer.decode([sampled_id])

    return dp_token

# Test sentence
sentence = "The cat sat on the mat."

dp_token = dp_masked_prediction(sentence)
print("DP rewritten token:", dp_token)
