import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import string
from collections import Counter

# (Long bench variants)
benchmarks = [
    "2wikimqa.jsonl",
    "multifieldqa_en.jsonl",
    "qasper.jsonl",
    "hotpotqa.jsonl",
]

def normalize_answer(s):
    """Lower, remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def remove_citations(text):
        return re.sub(r'\[\s*.*?\s*\]', '', text)

    return white_space_fix(remove_citations(remove_articles(remove_punc(lower(s)))))

def f1_score(prediction, ground_truth):
    """Compute F1 score from the WikiMQA script."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    # yes/no/noanswer special handling
    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    if(recall == 0 and precision == 0):
        return ZERO_METRIC
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall

def evaluate_single(pred, golds, context_length, benchmark):
    if not isinstance(golds, list):
        golds = [golds]

    best_f1 = 0.0
    best_p = 0.0
    best_r = 0.0

    for g in golds:
        f1, p, r = f1_score(pred, g)
        if f1 > best_f1:
            best_f1, best_p, best_r = f1, p, r

    return {
        "context_length": context_length,
        "benchmark": benchmark,
        "f1": best_f1,
        "precision": best_p,
        "recall": best_r,
        "total": 1,
    }

def construct_prompt_context(row):
    SYSTEM_PROMPT = 'Answer the question using only the provided passages.\n\n' 
    SYSTEM_PROMPT += 'Verify your answer directly against the text, and cite only the passages you used in your final answer. Cite passages in the form [Title].\n\n' 
    SYSTEM_PROMPT += "Respond in the following format: \n\n <reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n\n"

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question: {row['input']}\n\n"
        f"The following are given documents:\n"
        f"{row['context']}"
    )

if __name__ == "__main__":
    # Set up device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    print(f"Using device: {device} | dtype: {dtype}")

    # Load model, tokenizer
    print("Loading model and tokenizer")
    model_path = "mamba-1.4b-rag-rl-grpo-random/checkpoint-1600/"
    all_data = []

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left" # Mamba generation usually requires left padding
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype).to(device)
    model.eval()

    results_json_path = f"full_results_{model_path}.json"
    print(f'Starting {model_path}')
    for benchmark in benchmarks:
        metrics = {"f1": 0, "precision": 0, "recall": 0, "total": 0, "context_length": 0}
        with open(benchmark, 'r', encoding="utf-8") as f2:
            data = [json.loads(line) for line in f2]
        N = len(data)

        for d in tqdm(data, desc=f"Evaluating {benchmark}"):
            # Construct prompt
            prompt = construct_prompt_context(d)
            prompt_tokens = len(tok.tokenize(prompt))

            # Get model output
            gen_kwargs = dict(max_new_tokens=512, eos_token_id=tok.eos_token_id)
            inputs = tok(prompt, return_tensors="pt").to(device)
            input_length = inputs.input_ids.shape[1]

            with torch.no_grad():
                outs = model.generate(**inputs, **gen_kwargs)

            generated_tokens = outs[0][input_length:]
            answer = tok.decode(generated_tokens, skip_special_tokens=True)

            # Ground truth
            actual = d['answers']

            # Evaluate model output vs actual output
            eval_metrics = evaluate_single(answer, actual, prompt_tokens, benchmark)
            for k in metrics:
                metrics[k] += eval_metrics[k]
            all_data.append(eval_metrics)
            print(f"\nf1 on {benchmark} so far: {metrics['f1'] / metrics['total']}")

        # Finalize metrics for this benchmark
        for k in metrics:
            if k != "total":
                metrics[k] = metrics[k] / metrics["total"]
            print(f"{k} for {benchmark}: {metrics[k]}")

        with open(results_json_path, "w", encoding="utf-8") as f_json:
            json.dump(all_data, f_json, indent=4)

    print(f'finished {model_path}')