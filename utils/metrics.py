from collections import Counter
import re

import evaluate


BLEU = evaluate.load("bleu")


def normalize_text(text):
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text):
    return normalize_text(text).split()


def ngrams(tokens, n):
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def f1_score(overlap, pred_total, ref_total):
    if pred_total == 0 or ref_total == 0 or overlap == 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / ref_total
    return 2 * precision * recall / (precision + recall)


def rouge_n(prediction, reference, n):
    pred_ngrams = Counter(ngrams(tokenize(prediction), n))
    ref_ngrams = Counter(ngrams(tokenize(reference), n))
    overlap = sum((pred_ngrams & ref_ngrams).values())
    return f1_score(overlap, sum(pred_ngrams.values()), sum(ref_ngrams.values()))


def lcs_length(a, b):
    if not a or not b:
        return 0

    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l(prediction, reference):
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    overlap = lcs_length(pred_tokens, ref_tokens)
    return f1_score(overlap, len(pred_tokens), len(ref_tokens))


def compute_metrics(predictions, references):
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")

    if not predictions:
        return {
            "bleu": 0.0,
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "exact_match": 0.0,
            "normalized_exact_match": 0.0,
        }

    bleu = BLEU.compute(
        predictions=predictions,
        references=[[reference] for reference in references],
    )

    rouge1 = 0.0
    rouge2 = 0.0
    rouge_l_score = 0.0
    exact_matches = 0
    normalized_exact_matches = 0

    for prediction, reference in zip(predictions, references):
        rouge1 += rouge_n(prediction, reference, 1)
        rouge2 += rouge_n(prediction, reference, 2)
        rouge_l_score += rouge_l(prediction, reference)

        if prediction == reference:
            exact_matches += 1
        if normalize_text(prediction) == normalize_text(reference):
            normalized_exact_matches += 1

    total = len(predictions)
    return {
        "bleu": round(float(bleu["bleu"]), 4),
        "rouge1": round(rouge1 / total, 4),
        "rouge2": round(rouge2 / total, 4),
        "rougeL": round(rouge_l_score / total, 4),
        "exact_match": round(exact_matches / total, 4),
        "normalized_exact_match": round(normalized_exact_matches / total, 4),
    }
