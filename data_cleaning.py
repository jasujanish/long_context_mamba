import pandas as pd
import random
from tqdm import tqdm
import json 
import numpy as np

def create_combined():
    """
    Combine 2Wiki and HotpotQA training datasets, saves to pickle file
    """
    # Combine data
    print("Reading 2wiki_train.parquet")
    df1 = pd.read_parquet('2wiki_train.parquet')
    print("Reading hotpot_train.json")
    df2 = pd.read_json('hotpot_train.json')
    cols = ["answer", "context", "question", "supporting_facts"]
    df1 = df1[cols]
    df2 = df2[cols]
    combined = pd.concat([df1, df2], ignore_index=True)
    combined["id"] = combined.index
    
    print("Saving combined pickle")
    # Using compression='gzip' to keep file size manageable
    combined.to_pickle('combined_wiki_hotpot_train.pkl', compression='gzip') 
    print("Combined dataset saved")

def process_dataset_row(row):
    try:
        row_id = row['id']
        question = row['question']
        answer = row['answer']
        
        raw_context = row['context'] 
        supporting_facts = row['supporting_facts']
        
        # Edge case: If 2Wiki loaded as numpy array, convert to list for easier handling
        if isinstance(raw_context, np.ndarray): raw_context = raw_context.tolist()
        if isinstance(supporting_facts, np.ndarray): supporting_facts = supporting_facts.tolist()

        if not raw_context: return None

        # Create set for fast lookup
        gold_titles = set()
        if supporting_facts:
            # Handle case where supporting_facts might be None or empty
            for fact in supporting_facts:
                if len(fact) > 0:
                    gold_titles.add(fact[0])
        
        gold_passages = []
        distractor_passages = []
        
        for item in raw_context:
            if len(item) < 2: continue
            title = item[0]
            content_val = item[1]
            # content_val might be a list of sentences or a single string
            if isinstance(content_val, list):
                text = " ".join([str(s) for s in content_val])
            else:
                text = str(content_val)
            
            passage_obj = {"title": title, "text": text}
            
            if title in gold_titles:
                gold_passages.append(passage_obj)
            else:
                distractor_passages.append(passage_obj)
        return {
            "id": row_id,
            "question": question,
            "answer": answer,
            "gold_passages": gold_passages,             
            "distractor_passages": distractor_passages 
        }
    except Exception as e:
        print(f"Error processing row {row.get('id', 'unknown')}: {e}")
        return None

def create_rag_rl_dataset(sample_percent=0.01):
    print("Loading combined dataset from Pickle")
    df = pd.read_pickle('combined_wiki_hotpot_train.pkl', compression='gzip')
    
    df = df.sample(frac=sample_percent, random_state=711).reset_index(drop=True)

    processed_rows = []
    print(f"Processing {len(df)} rows...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        proc = process_dataset_row(row)
        if proc:
            processed_rows.append(proc)

    if not processed_rows:
        print("No rows processed successfully.")
        return
        
    final_df = pd.DataFrame(processed_rows)
    print(f"Final dataset size: {len(final_df)} samples")
    
    # dump to pickle
    final_df['gold_passages'] = final_df['gold_passages'].apply(json.dumps)
    final_df['distractor_passages'] = final_df['distractor_passages'].apply(json.dumps)
    output_filename = 'rag_rl_training_data.pkl'
    final_df.to_pickle(output_filename, compression='gzip')
    print(f"Saved to {output_filename}")

if __name__ == "__main__":
    create_combined()
    create_rag_rl_dataset(sample_percent=1)