# Early re-implementation (kept for the record)

These scripts were my first attempt to implement DP-MLM from the paper's description, before switching to the
official `dpmlm` package. They are **not** used for any reported result. Differences from the paper
(discussed in Section 2.1 of the report):

1. Logits clipped to hand-picked bounds (±5) instead of calibrated RoBERTa bounds.
2. Temperature `(max - min) / eps`, missing the factor 2 of the exponential mechanism (actual guarantee 2·eps).
3. Only one token or one random 3-token span rewritten, instead of every non-stopword token.
4. TF-IDF cosine similarity instead of sentence-embedding cosine similarity.
