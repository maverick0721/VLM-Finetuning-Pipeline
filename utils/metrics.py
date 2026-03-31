import evaluate


bleu = evaluate.load("bleu")


def compute_metrics(predictions, references):
    score = bleu.compute(predictions=predictions, references=references)
    return {"bleu": score["bleu"]}
