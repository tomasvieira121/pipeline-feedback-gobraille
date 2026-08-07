import logging
from bert_score import score

# Suppress verbose transformers logging
logging.getLogger("transformers").setLevel(logging.ERROR)

def calculate_bertscore(predictions, references, lang="en"):
    """
    Calculates BERTScore for a list of predictions against references.
    
    Args:
        predictions (list of str): The generated responses.
        references (list of str): The ground truth responses.
        lang (str): Language for the model.
        
    Returns:
        dict: A dictionary containing precision, recall, f1 for each pair, 
              as well as the average across all pairs.
    """
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
