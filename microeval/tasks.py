from sklearn.metrics import f1_score, accuracy_score
import sacrebleu
from rouge_score import rouge_scorer

def eval_classification(preds, refs):
    """
    preds: list of predicted labels
    refs: list of true labels
    """
    return {
        "accuracy": accuracy_score(refs, preds),
        "f1": f1_score(refs, preds, average="weighted")
    }

def eval_translation(preds, refs):
    """
    preds: list of strings
    refs: list of strings
    """
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return {"bleu": bleu.score}

def eval_summarization(preds, refs):
    """
    preds: list of strings
    refs: list of strings
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = [scorer.score(r, p) for r, p in zip(refs, preds)]
    avg_scores = {k: sum(s[k].fmeasure for s in scores) / len(scores) for k in scores[0]}
    return avg_scores
