import pandas as pd
import random
from tqdm import tqdm
import json 
import numpy as np

def create_combined():
    """
    Combine Wikipedia and HotpotQA training datasets, save to parquet
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

    # We serialize everything to strings to ensure compatibility with parquet format
    print("Serializing complex columns to JSON strings")
    combined['context'] = combined['context'].apply(lambda x: json.dumps(x) if isinstance(x, list) else json.dumps([]))
    def safe_serialize(x):
        if x is None: return json.dumps([])
        if isinstance(x, (list, np.ndarray)): return json.dumps(list(x))
        return json.dumps([])
    combined['supporting_facts'] = combined['supporting_facts'].apply(safe_serialize)

    # We save the serialized combined dataset
    print("Saving combined parquet")
    combined.to_parquet('combined_wiki_hotpot_train.parquet', index=False)
    print("Combined dataset saved")

def process_dataset_row(row):
    """
    Parses a row from HotpotQA/2Wiki to separate Gold vs. Distractor passages into dict format
    """
    try:
        # Extract raw data
        row_id = row['id']
        question = row['question']
        answer = row['answer']
        
        # Raw context and supporting_facts are loaded as STRINGS now, so we deserialize
        raw_context = row['context']  # List of [title, [sentences]]
        supporting_facts = row['supporting_facts']
        if isinstance(raw_context, str): raw_context = json.loads(raw_context)
        if isinstance(supporting_facts, str): supporting_facts = json.loads(supporting_facts)
        if not raw_context: return None

        gold_titles = set([fact[0] for fact in supporting_facts]) if supporting_facts else set()
        
        # Clean context
        gold_passages = []
        distractor_passages = []
        for item in raw_context:
            # Format should be [title, [sentences]]
            if len(item) < 2: continue
            
            title = item[0]
            content_val = item[1]
            if isinstance(content_val, list):
                text = " ".join([str(s) for s in content_val])
            else:
                text = str(content_val)
            
            passage_obj = {
                "title": title,
                "text": text
            }
            
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
    """ 
    Creates a structured dataset for RAG-RL experiments. 
    Outputs a parquet file where 'gold' and 'distractor' contexts are separated.
    """
    # Load combined data
    print("Loading combined dataset")
    df = pd.read_parquet('combined_wiki_hotpot_train.parquet')
    df = df.sample(frac=sample_percent, random_state=711).reset_index(drop=True)

    # Process Rows
    processed_rows = []
    print(f"Processing {len(df)} rows...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        proc = process_dataset_row(row)
        if proc:
            processed_rows.append(proc)

    # Create Final DataFrame and Save
    if not processed_rows:
        print("No rows processed successfully.")
        return
    final_df = pd.DataFrame(processed_rows)
    print(f"Final dataset size: {len(final_df)} samples")
    
    # We serialize the output lists as well to ensure the final parquet is clean and portable
    final_df['gold_passages'] = final_df['gold_passages'].apply(json.dumps)
    final_df['distractor_passages'] = final_df['distractor_passages'].apply(json.dumps)

    output_filename = 'rag_rl_training_data.parquet'
    final_df.to_parquet(output_filename, index=False)
    print(f"Saved to {output_filename}")

if __name__ == "__main__":
    create_combined()
    create_rag_rl_dataset(sample_percent=0.05) 