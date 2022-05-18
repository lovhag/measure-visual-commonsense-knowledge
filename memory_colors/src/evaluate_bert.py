from transformers import BertTokenizer, BertForMaskedLM
from typing import Optional, List, Dict
import torch
import numpy as np

def evaluate_zs_classification(
    examples: List[Dict], 
    class_names: List[str], 
    bert_model: str, 
    k: int, 
    question_template: str,
    tokenizer_name_or_path: str = "bert-base-uncased"
):
    """Evaluates BERT on a dataset of commonsense object color questions

    Args:
        examples (List[Dict]):
            required keys: 
            "item" - to replace [ITEM] in question_template, 
            "label" - to replace [MASK] in question_template,
            "descriptor" - to replace [DESCRIPTOR] in question_template
        class_names (List[str]): List of class names
        bert_model (str): Name of pretrained huggingface transformers BERT model to load
        k (int): Evaluate precision@k
        question_template (str): Ex "The color of [DESCRIPTOR] [ITEM] is [MASK].". Must contain [ITEM], [MASK] and [DESCRIPTOR]. [DESCRIPTOR] can for example be "a"/"an"/"the animal".
        tokenizer_name_or_path (str): Name or path of the tokenizer to be used
    """
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name_or_path)
    model = BertForMaskedLM.from_pretrained(bert_model)

    class_names_ids = names_to_token_ids(class_names, tokenizer)
    
    questions = []
    for ex in examples:
        if ex["descriptor"] == '':
            question = question_template.replace("[DESCRIPTOR] ", "").replace("[ITEM]", ex["item"])
        else:
            question = question_template.replace("[DESCRIPTOR]", ex["descriptor"]).replace("[ITEM]", ex["item"])
        questions.append(question)
        
    inputs = tokenizer(questions, return_tensors="pt", padding=True)
    pred = predict_masked(inputs, tokenizer, model, class_names_ids)
    topk = pred.topk(k).indices # Rank color tokens and select top k (batch, k)
    gt = torch.tensor([class_names.index(ex["label"]) for ex in examples]).unsqueeze(1) # Index of ground truth color - (batch, 1)
    
    precision_at_k = np.true_divide((topk == gt).sum(), gt.shape[0])

    return float(precision_at_k)


def names_to_token_ids(names: List[str], tokenizer):
    names_ids = tokenizer(names, add_special_tokens=False).input_ids
    names_ids = [id for ids in names_ids for id in ids]
    assert len(names_ids) == len(names), "ERROR: One or more class_names was tokenized to multiple tokens"
    return names_ids


def predict_masked(inputs, 
                   tokenizer: BertTokenizer, 
                   model: BertForMaskedLM, 
                   filter_token_ids: Optional[List]=None) -> torch.Tensor:
    mask_idx = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=False)[:, 1][:, None]
    assert len(mask_idx) == inputs.input_ids.shape[0], "ERROR: Found multiple [MASK] tokens per example"

    outputs = model(**inputs)
    
    pred = outputs[0][:, :, filter_token_ids] if filter_token_ids is not None else outputs[0] # Select color tokens only - (batch, seq, #vocab)
    pred = pred.gather(1, mask_idx.repeat(1, pred.shape[-1]).unsqueeze(1)).squeeze(1) # Select the mask token for each sequence - (batch, #colors)
    return pred


if __name__ == "__main__":
    import argparse
    import sys
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=sys.stdin, type=argparse.FileType('r', encoding="utf-8"), help="Json input (must have \"item\", \"label\" and \"descriptor\")")
    parser.add_argument("--question-template")
    parser.add_argument("--bert-model", default="bert-base-uncased")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--k", default=1, type=int)
    args = parser.parse_args()

    examples = [json.loads(line) for line in args.data.readlines()]
    labels = list(set(ex["label"] for ex in examples))

    precision_at_k = evaluate_zs_classification(
        examples=examples,
        class_names=labels,
        bert_model=args.bert_model, 
        tokenizer_name_or_path=args.tokenizer or args.bert_model,
        k=args.k, 
        question_template=args.question_template
    )

    print(precision_at_k)
