from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def tokenize(text):
    return text.split()

def compute_bleu(original, rewritten):
    ref = tokenize(original)
    hyp = tokenize(rewritten)
    smoothie = SmoothingFunction().method4
    return sentence_bleu([ref], hyp, smoothing_function=smoothie)

def compute_cosine_similarity(original, rewritten):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([original, rewritten])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

if __name__ == "__main__":
    original = "John lives in Sydney and works at a bank."
    rewritten = "ProfessorI wells Downtown Sheffield so engineers and an developer website Pic"

    bleu = compute_bleu(original, rewritten)
    cs = compute_cosine_similarity(original, rewritten)

    print("BLEU:", bleu)
    print("Cosine similarity:", cs)
