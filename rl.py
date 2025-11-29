import os
from typing import List, Optional
from rewards import reward_answer_correctness, reward_citation_accuracy, reward_formatting, reward_repetition_penalty
import transformers
os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "1"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import GRPOTrainer, GRPOConfig
import json
import pandas as pd
from datasets import Dataset

# scaling constants
GAMMA_ANSWER = 6.0
GAMMA_CIT_CORRECT = 3.0
GAMMA_FORMAT = 2.0
GAMMA_REP = 1.0

# Max prompt length
MAX_PROMPT_LENGTH = 8192

DEVICE_MAP = {
    "backbone.embeddings": 1,

    "backbone.layers.0": 1,
    "backbone.layers.1": 1,
    "backbone.layers.2": 1,
    "backbone.layers.3": 1,
    "backbone.layers.4": 1,
    "backbone.layers.5": 1,
    "backbone.layers.6": 1,
    "backbone.layers.7": 1,

    "backbone.layers.8": 2,
    "backbone.layers.9": 2,
    "backbone.layers.10": 2,
    "backbone.layers.11": 2,
    "backbone.layers.12": 2,
    "backbone.layers.13": 2,
    "backbone.layers.14": 2,
    "backbone.layers.15": 2,

    "backbone.layers.16": 3,
    "backbone.layers.17": 3,
    "backbone.layers.18": 3,
    "backbone.layers.19": 3,
    "backbone.layers.20": 3,
    "backbone.layers.21": 3,
    "backbone.layers.22": 3,
    "backbone.layers.23": 3,

    "backbone.layers.24": 4,
    "backbone.layers.25": 4,
    "backbone.layers.26": 4,
    "backbone.layers.27": 4,
    "backbone.layers.28": 4,
    "backbone.layers.29": 4,
    "backbone.layers.30": 4,
    "backbone.layers.31": 4,

    "backbone.layers.32": 5,
    "backbone.layers.33": 5,
    "backbone.layers.34": 5,
    "backbone.layers.35": 5,
    "backbone.layers.36": 5,
    "backbone.layers.37": 5,
    "backbone.layers.38": 5,
    "backbone.layers.39": 5,

    "backbone.layers.40": 6,
    "backbone.layers.41": 6,
    "backbone.layers.42": 6,
    "backbone.layers.43": 6,
    "backbone.layers.44": 6,
    "backbone.layers.45": 6,
    "backbone.layers.46": 6,
    "backbone.layers.47": 6,

    "backbone.layers.48": 7,
    "backbone.layers.49": 7,
    "backbone.layers.50": 7,
    "backbone.layers.51": 7,
    "backbone.layers.52": 7,
    "backbone.layers.53": 7,
    "backbone.layers.54": 7,
    "backbone.layers.55": 7,

    "backbone.norm_f": 7,
    "lm_head": 7
}
# Reward wrappers
def answer_reward(completions: List[str], answer: List[str], **kwargs,) -> List[float]:
    """Wraps answer reward with scaling."""
    base = reward_answer_correctness(completions, answer)
    return [GAMMA_ANSWER * float(x) for x in base]

def citation_reward(completions: List[str], gold_ids: List[List[str]], **kwargs,) -> List[float]:
    """Wraps citation reward with scaling."""
    base = reward_citation_accuracy(completions, gold_ids)
    return [GAMMA_CIT_CORRECT * float(x) for x in base]

def formatting_reward(completions: List[str], **kwargs,) -> List[float]:
    """Wraps formatting reward with scaling."""
    base = reward_formatting(completions)
    return [GAMMA_FORMAT * float(x) for x in base]

def repetition_reward(completions: List[str], **kwargs,) -> List[float]:
    """Wraps repetition reward with scaling."""
    base = reward_repetition_penalty(completions)
    return [GAMMA_REP * float(x) for x in base]

MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"

SYSTEM_PROMPT = 'Answer the question using only the provided passages.\n\n' 
SYSTEM_PROMPT += 'Verify your answer directly against the text, and cite only the passages you used in your final answer. Cite passages in the form [Title].\n\n' 
SYSTEM_PROMPT += "Respond in the following format: \n\n <reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n\n"

class LogToFileCallback(TrainerCallback):
    def __init__(self, output_file):
        self.output_file = output_file
        # Create/Clear the file when we start
        with open(self.output_file, "w") as f:
            f.write("Training Logs\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        # This function runs whenever the trainer logs metrics (logging_steps)
        if logs:
            with open(self.output_file, "a") as f:
                f.write(json.dumps(logs) + "\n")
                
def sort_by_num_distractors(df: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    """
    Curriculum sort: order examples by number of distractor passages.

    Assumes `distractor_passages` is a JSON-serialized list of passages.
    """
    def _count_distractors(s: str) -> int:
        try:
            passages = json.loads(s)
            if isinstance(passages, list):
                return len(passages)
            return 0
        except Exception:
            return 0

    df = df.copy()
    df["num_distractors"] = df["distractor_passages"].apply(_count_distractors)
    df = df.sort_values("num_distractors", ascending=ascending).reset_index(drop=True)
    df = df.drop(columns=["num_distractors"])
    return df

def sort_random(df: pd.DataFrame) -> pd.DataFrame:
    """
    Randomly shuffle the examples (no curriculum).
    """
    return df.sample(frac=1.0, random_state=711).reset_index(drop=True)

def build_reference(row):
    """
    Builds the reference text from the gold and distractor passages
    """    
    gold = json.loads(row["gold_passages"])
    distractors = json.loads(row["distractor_passages"])

    docs = []
    # we want titles to be the citation keys
    for p in gold + distractors:
        title = p["title"]
        text = p["text"]
        docs.append(f"Document {title}\n{text}")

    return "\n\n".join(docs)

def make_grpo_dataset(pickle_path: str, tokenizer: AutoTokenizer, max_rows: Optional[int] = None, curriculum: bool = True, sample: Optional[float] = None) -> Dataset:
    # Load in DF, sort based on curriculum
    df = pd.read_pickle(pickle_path, compression='gzip')
    if max_rows is not None:
        df = df.head(max_rows)
    if sample is not None:
        df = df.sample(frac=sample, random_state=711).reset_index(drop=True)
    if curriculum:
        df = sort_by_num_distractors(df, ascending=True)
    else:
        df = sort_random(df)

    # Build reference text
    df["reference"] = df.apply(build_reference, axis=1)

    # Gold IDs = titles of gold passages (for citation reward)
    df["gold_ids"] = df["gold_passages"].apply(
        lambda s: [p["title"] for p in json.loads(s)]
    )

    # Prompt expected by GRPOTrainer
    def _build_prompt(row):
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Question: {row['question']}\n\n"
            f"The following are given documents:\n"
            f"{row['reference']}"
        )

    df["prompt"] = df.apply(_build_prompt, axis=1)

    print(f"Dataset size before filtering: {len(df)}")
    
    # Calculate token lengths
    # We use a simple lambda with the tokenizer. 
    # Note: We assume truncation=False here to get the real length.
    df["token_len"] = df["prompt"].apply(lambda x: len(tokenizer(x, add_special_tokens=False)["input_ids"]))
    
    # Filter: Keep only rows where token_len <= max_prompt_length
    df = df[df["token_len"] <= MAX_PROMPT_LENGTH]
    
    print(f"Dataset size after filtering (> {MAX_PROMPT_LENGTH} removed): {len(df)}")

    # Only keep columns we need
    df = df[["prompt", "answer", "gold_ids"]]

    return Dataset.from_pandas(df, preserve_index=False)

def main():
    # Tokenizer
    print('loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load data
    print('loading data')
    train_dataset = make_grpo_dataset(
        pickle_path="rag_rl_training_data.pkl",
        tokenizer=tokenizer,
        max_rows=None, 
        sample=0.01, # taking a 0.01% sample 
    )

    # Safe dtype selection
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if use_bf16:
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    # manually dispatch model to avoid OOM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=DEVICE_MAP,
        trust_remote_code=True,
    )

    if hasattr(model, "config"):
        model.config.use_cache = False
        model.generation_config.use_cache = True

    if hasattr(model.config, "architectures") and model.config.architectures:
        cls_name = model.config.architectures[0]  # e.g. "NemotronHForCausalLM"
        if not hasattr(transformers, cls_name):
            setattr(transformers, cls_name, model.__class__)
            
    # GRPO config
    training_args = GRPOConfig(
        output_dir="/data/user_data/jiaruiwu/11711/Final/nemotron-rag-rl-grpo",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        logging_steps=100,
        save_steps=200,
        save_total_limit=3,
        bf16=use_bf16,
        gradient_checkpointing=True,
        num_generations=4,                   # completions per prompt
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=256,
        beta=0.01,
        scale_rewards="batch",          # generally more stable than per-group
        disable_dropout=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.6,
    )

    # GRPO trainer
    print('starting trainer')
    file_logger = LogToFileCallback("training_logs.txt")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=[answer_reward, citation_reward, formatting_reward, repetition_reward],
        callbacks=[file_logger],
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
if __name__ == "__main__":
    main()