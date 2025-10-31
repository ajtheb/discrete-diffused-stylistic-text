import pandas as pd
from transformers import GPT2TokenizerFast

# Initialize tokenizer (choose your model's tokenizer)
tokenizer = GPT2TokenizerFast.from_pretrained('/home/models/gpt2')
tokenizer.pad_token = '<PAD>'

def get_max_context_length(files):
    max_length = 0
    for file in files:
        df = pd.read_csv(file)
        
        # Combine all columns into full context
        df["full_context"] = df.apply(
            lambda row: f"{row['Hatespeech']} [SEP] {row['Style']} [SEP] {row['Target']} [SEP] {row['Counterspeech']}", 
            axis=1
        )
        
        # Calculate token counts
        token_counts = df["full_context"].apply(
            lambda x: len(tokenizer.encode(x, add_special_tokens=False))
        )
        
        current_max = token_counts.max()
        if current_max > max_length:
            max_length = current_max
            
    return max_length

# Usage
files = ["Train.csv", "Eval.csv", "Test.csv"]
max_context_length = get_max_context_length(files)
print(f"Maximum context length: {max_context_length} tokens")
