import torch
from transformers import RobertaTokenizerFast, RobertaForMaskedLM

EPSILON = 500
CLIP_MIN = -5
CLIP_MAX = 5

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
model = RobertaForMaskedLM.from_pretrained("roberta-base")
model.eval()

def dp_sample(logits, epsilon):
    logits = torch.clamp(logits, CLIP_MIN, CLIP_MAX)
    T = (CLIP_MAX - CLIP_MIN) / epsilon
    probs = torch.softmax(logits / T, dim=-1)
    return torch.multinomial(probs, 1).item()

def dp_rewrite_span(sentence, span_size=3, epsilon=EPSILON):
    tokens = tokenizer.tokenize(sentence)
    n = len(tokens)

    # Choose a random span to DP-rewrite
    start = torch.randint(0, n - span_size, (1,)).item()
    end = start + span_size

    masked_tokens = tokens.copy()
    for i in range(start, end):
        masked_tokens[i] = tokenizer.mask_token

    combined = f"{sentence} </s> {' '.join(masked_tokens)}"
    enc = tokenizer(combined, return_tensors="pt")
    input_ids = enc["input_ids"][0]

    sep_index = (input_ids == tokenizer.sep_token_id).nonzero()[0].item()

    rewritten = tokens.copy()

    for i in range(start, end):
        mask_pos = sep_index + 1 + i
        with torch.no_grad():
            logits = model(input_ids.unsqueeze(0)).logits[0][mask_pos]
        sampled_id = dp_sample(logits, epsilon)
        rewritten[i] = tokenizer.decode([sampled_id])

    return " ".join(rewritten)

if __name__ == "__main__":
    sent = "John lives in Sydney and works at a bank."
    print("Original:", sent)
    print("DP rewritten:", dp_rewrite_span(sent))
