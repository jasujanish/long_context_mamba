import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

# --- CONFIGURATION ---
# UPDATED: Matches the model used in train_mamba.py
BASE_MODEL_ID = "state-spaces/mamba-1.4b-hf"
RL_MODEL_PATH = "mamba-1.4b-rag-rl-grpo" 

# A prompt that requires citations to test behavior
TEST_PROMPT = (
    "Answer the question using only the provided passages.\n\n"
    "Verify your answer directly against the text, and cite only the passages you used in your final answer. "
    "Cite passages in the form [Title].\n\n"
    "Respond in the following format: \n\n <reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n\n"
    "Question: What is the primary theme of the document?\n\n"
    "The following are given documents:\n"
    "Document [The Great Gatsby]\nThe green light at the end of the dock represents Gatsby's hopes and dreams for the future.\n\n"
    "Document [Moby Dick]\nThe white whale represents the chaotic and indifferent nature of the universe."
)

def register_custom_architecture(model):
    """Registers the architecture to transformers if missing (crucial for some Mamba versions)."""
    if hasattr(model.config, "architectures") and model.config.architectures:
        cls_name = model.config.architectures[0]
        if not hasattr(transformers, cls_name):
            print(f"  -> Registering custom class: {cls_name}")
            setattr(transformers, cls_name, model.__class__)

def generate_response(model_path, model_name_for_display):
    print(f"\n[{model_name_for_display}] Loading from {model_path}...")
    
    # Load Tokenizer
    # trust_remote_code=True is often needed for Mamba architectures
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Mamba generation usually requires left padding
    tokenizer.padding_side = "left"
        
    # Load Model
    # We use float16 or bfloat16 to save memory
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading {model_name_for_display}: {e}")
        return

    # Architecture Hack
    register_custom_architecture(model)

    # Generate
    inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device)
    
    print(f"[{model_name_for_display}] Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, 
            do_sample=False, # Deterministic for testing
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Strip the prompt from the output for cleaner viewing
    # Mamba sometimes echoes the prompt differently, so we robustly strip based on length
    response_only = response[len(TEST_PROMPT):]
    
    # Clean up VRAM immediately
    del model
    del inputs
    del outputs
    torch.cuda.empty_cache()
    gc.collect()
    
    return response_only

def main():
    print("--- STARTING COMPARISON ---")
    
    # 1. Run Base Model
    base_response = generate_response(BASE_MODEL_ID, "BASE MODEL")
    
    # 2. Run RL Model
    rl_response = generate_response(RL_MODEL_PATH, "RL TRAINED MODEL")
    
    print("\n" + "="*80)
    print(f"PROMPT:\n{TEST_PROMPT}")
    print("="*80)
    
    print(f"\n>>> BASE MODEL RESPONSE:\n{base_response}")
    print("-" * 80)
    
    print(f"\n>>> RL MODEL RESPONSE:\n{rl_response}")
    print("="*80)

if __name__ == "__main__":
    main()