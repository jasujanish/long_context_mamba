import os
from typing import List, Optional
from rewards import reward_answer_correctness, reward_citation_accuracy, reward_formatting
import transformers
os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "1"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
import json
import pandas as pd
from datasets import Dataset

# scaling constants
GAMMA_ANSWER = 5.0
GAMMA_CIT_CORRECT = 3.0
GAMMA_FORMAT = 1.0


def answer_reward(completions: List[str], answer: List[str], **kwargs,) -> List[float]:
    """
    Wraps answer reward with scaling.

    GRPOTrainer will call this as:
      answer_reward(prompts=..., completions=..., answer=..., ...)
    We only care about completions + answer here.
    """
    base = reward_answer_correctness(completions, answer)
    return [GAMMA_ANSWER * float(x) for x in base]

def citation_reward(completions: List[str], gold_ids: List[List[str]], **kwargs,) -> List[float]:
    """
    Wraps citation reward with scaling; assumes completions cite [Title]
    and gold_ids is a list of lists of titles.
    """
    base = reward_citation_accuracy(completions, gold_ids)
    return [GAMMA_CIT_CORRECT * float(x) for x in base]

def formatting_reward(completions: List[str], **kwargs,) -> List[float]:
    """Wraps formatting reward with scaling."""
    base = reward_formatting(completions)
    return [GAMMA_FORMAT * float(x) for x in base]

MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"

SYSTEM_PROMPT = 'Answer the question using only the provided passages.\n\n' 
SYSTEM_PROMPT += 'Verify your answer directly against the text, and cite only the passages you used in your final answer. Cite passages in the form [Title].\n\n' 
SYSTEM_PROMPT += "Respond in the following format: \n\n <reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n\n"

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

def make_grpo_dataset(pickle_path: str, max_rows: Optional[int] = None, curriculum: bool = True,) -> Dataset:
    # Load in DF, sort based on curriculum
    df = pd.read_pickle(pickle_path, compression='gzip')
    if max_rows is not None:
        df = df.head(max_rows)
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

    # Only keep columns we need, GRPOTrainer should have "prompt"
    df = df[["prompt", "answer", "gold_ids"]]

    return Dataset.from_pandas(df, preserve_index=False)

def main():
    # Load data
    print('loading data')
    train_dataset = make_grpo_dataset(
        "rag_rl_training_data.pkl",
        max_rows=None,  # or small slice while debugging
    )

    # Tokenizer and model
    print('loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Safe dtype selection
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if use_bf16:
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # should save compute
    if hasattr(model, "config"):
        model.config.use_cache = False


    if hasattr(model.config, "architectures") and model.config.architectures:
        cls_name = model.config.architectures[0]  # e.g. "NemotronHForCausalLM"
        if not hasattr(transformers, cls_name):
            setattr(transformers, cls_name, model.__class__)

    # GRPO config
    # training_args = GRPOConfig(
    #     output_dir="nemotron-rag-rl-grpo",
    #     num_train_epochs=1,
    #     per_device_train_batch_size=1,  # bump if VRAM allows
    #     gradient_accumulation_steps=4,
    #     learning_rate=5e-6,
    #     logging_steps=10,
    #     save_steps=200,
    #     save_total_limit=3,
    #     bf16=use_bf16,
    #     gradient_checkpointing=True,
    #     group_size=4,                   # completions per prompt
    #     max_prompt_length=4096,
    #     max_completion_length=512,
    #     kl_coef=0.01,
    #     scale_rewards="batch",          # generally more stable than per-group
    #     disable_dropout=True,
    # )
    training_args = GRPOConfig(
        output_dir="nemotron-rag-rl-grpo",
        num_train_epochs=1,
        per_device_train_batch_size=1,   # bump if VRAM allows
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        bf16=use_bf16,                   # or fp16=True if you prefer
        gradient_checkpointing=True,

        # >>> key fixes here <<<
        num_generations=4,               # was: group_size=4
        max_prompt_length=4096,
        max_completion_length=512,
        beta=0.01,                       # was: kl_coef=0.01
        scale_rewards="batch",           # "group", "batch", or "none"
        disable_dropout=True,
    )

    print('starting trainer')
    # GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=[answer_reward, citation_reward, formatting_reward],
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()