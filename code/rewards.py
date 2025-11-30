import re
import json
import random
import pandas as pd
from typing import List, Dict, Any
from collections import Counter
import string

def get_citations(text: str) -> List[str]:
    """Extracts citations like [1], [chill], [ chill ] from text."""
    matches = re.findall(r'\[\s*.*?\s*\]', text)
    return list(set(m for m in matches))

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

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    """Compute F1 score"""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    # yes/no/noanswer special handling
    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    if recall == 0 or precision == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1
    
def reward_answer_correctness(completions: List[str], answers: List[str]) -> List[float]:
    """
    R_ans = { 1.0 if exact match between completition and answer
            { F1 score else
    Rewards correct answers
    """
    rewards = []
    for completion, ground_truth in zip(completions, answers):
        completion_clean = re.sub(r'\[\s*.*?\s*\]', '', completion).strip().lower()
        ground_truth_clean = ground_truth.strip().lower()
        
        # Exact match reward
        if ground_truth_clean in completion_clean:
            rewards.append(1.0)
        # F1 score reward
        else:
            overlap = len(set(completion_clean.split()) & set(ground_truth_clean.split()))
            rewards.append(f1_score(completion_clean, ground_truth_clean)) 
    return rewards

def reward_citation_accuracy(completions: List[str], gold_ids: List[List[str]]) -> List[float]:
    """
    R_cit = f1 score between cited and gold ids
    Rewards  citing gold docs, penalizes citing distractors and hallucinations.
    """
    rewards = []
    for completion, correct_ids in zip(completions, gold_ids):
        predicted_ids = get_citations(completion)
        correct_set = set(correct_ids)
        predicted_set = set(predicted_ids)
        
        # if no gold docs, reward no citations
        if not correct_set:
            rewards.append(1.0 if not predicted_set else 0.0)
            continue
            
        # Recall (gold docs cited)
        recall = len(predicted_set.intersection(correct_set)) / len(correct_set)
        
        # Precision (only gold docs cited)
        precision = 1.0
        if len(predicted_set) > 0:
            precision = len(predicted_set.intersection(correct_set)) / len(predicted_set)
        
        # Combined F1-style score for citation
        if recall == 0 or precision == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        rewards.append(f1)
        
    return rewards

def reward_repetition_penalty(completions: List[str]) -> List[float]:
    """
    R_rep = { -1                                if no lines at all
            { 0                                 if no repeated lines
            { max(-(num repeated lines), -5.0)  else
    Returns a penalty of -(num repeated lines) if any repeated lines are found in the completion.
    """
    rewards = []
    for completion in completions:
        lines = [line.strip() for line in completion.split('\n') if line.strip()]
        if not lines:
            rewards.append(-1.0) 
            continue
        line_set = set(lines)
        num_dupes = len(lines) - len(line_set)
        if num_dupes == 0:
            rewards.append(0.0)
        else:
            rewards.append(max(-1.0 * num_dupes, -5.0))        
    return rewards

def _is_well_formatted(text: str,) -> bool:
    """
    Check if the completion follows the required XML-style format
    Returns:
        -1.0 if completely incorrect
        -0.5 if tags are present
        0 if tags are in correct order
        +0.5 if perfectly formatted (no extra text)
    """
    reward = -1.0
    if "<reasoning>" in text and "</reasoning>" and "<answer>" in text and "</answer>" not in text:
        reward+=0.5
        r_open = text.find("<reasoning>")   
        r_close = text.find("</reasoning>")
        a_open = text.find("<answer>")
        a_close = text.find("</answer>")
        if 0 <= r_open < r_close < a_open < a_close:
            reward += 0.5
            pattern = r'^\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$'
            if re.fullmatch(pattern, text, flags=re.DOTALL):
                reward += 0.5
    return reward

def reward_formatting(completions: List[str]) -> List[float]:
    """
    R_formatting =  { +0.5  if perfectly formatted (no extra text)
                    { 0     if tags are in correct order
                    { -0.5  if tags are present
                    { -1.0  if completely incorrect

    Formatting is defined by:
      - Correct XML-style tags: <reasoning>...</reasoning><answer>...</answer>
      - Not excessively long (> max_chars)
    """
    rewards = []
    for completion in completions:
        rewards.append(_is_well_formatted(completion))
    return rewards
