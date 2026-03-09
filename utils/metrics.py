import evaluate

bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")


def compute_metrics(predictions, references):

    bleu = bleu_metric.compute(
        predictions=predictions,
        references=[[ref] for ref in references]
    )

    rouge = rouge_metric.compute(
        predictions=predictions,
        references=references
    )

    return {
        "bleu": bleu["bleu"],
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"]
    }