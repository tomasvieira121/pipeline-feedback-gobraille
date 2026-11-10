import logging
from bert_score import score

# Suppress verbose transformers logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# calculates bertscore for predictions against references
def calculate_bertscore(predictions, references, lang="en"):
    if not predictions or not references:
        return {}
    
    P, R, F1 = score(predictions, references, lang=lang, verbose=False)
    
    return {
        "precision": P.tolist(),
        "recall": R.tolist(),
        "f1": F1.tolist(),
        "avg_precision": P.mean().item(),
        "avg_recall": R.mean().item(),
        "avg_f1": F1.mean().item()
    }
