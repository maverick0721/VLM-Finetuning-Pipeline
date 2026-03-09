from datasets import load_metric


bleu = load_metric("bleu")


def compute_metrics(predictions, references):

    score = bleu.compute(
        predictions=predictions,
        references=references
    )

    return {
        "bleu": score["bleu"]
    }