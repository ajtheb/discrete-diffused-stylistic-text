import transformers
import torch
import pandas as pd
import argparse
from time import time
from tqdm import tqdm

def load_model(model_id):
    """Loads the GPT-2 Large model and tokenizer."""
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    
    pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id
    pipeline.tokenizer.padding_side = 'left'
    
    return tokenizer, pipeline

def get_prompt(instruction):
    """Formats the instruction prompt."""
    return f"{instruction}"

def generate_prompts(df):
    """Generates prompts for counterspeech generation."""
    prompts = []
    for ind in tqdm(df.index, desc="Processing rows"):
        hatespeech = df.loc[ind, 'Hatespeech']
        style = df.loc[ind, 'Style']
        leader = 'Mahatma Gandhi' if style == 'Gandhi' else 'Nelson Mandela'
        
        prompt = (f"You are {leader}. Respond to the following hate speech with a counterspeech rooted in "
                  f"nonviolence, truth, peace, and understanding. Use one of your quotes if appropriate.\n\n"
                  f"Hate speech: {hatespeech}\n\nCounterspeech:")
        
        prompts.append(get_prompt(prompt))
    return prompts

def test_model(tokenizer, pipeline, prompts, batch_size=32):
    """Runs the model on input prompts and returns generated responses."""
    start_time = time()
    sequences = pipeline(
        prompts,
        do_sample=False,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        batch_size=batch_size
    )
    elapsed_time = round(time() - start_time, 3)
    print(f"Test inference time: {elapsed_time} sec.")
    
    return [seq[0]['generated_text'].split("Counterspeech:")[1].strip() for seq in sequences]

def main():
    parser = argparse.ArgumentParser(description="Generate counterspeech using GPT-2 Large")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--prompt_type", type=str, required=True, help="Prompt type")
    parser.add_argument("--model_path", type=str, default="/home/models/gpt2-large", help="Path to the model")
    args = parser.parse_args()
    
    prompt_type = args.prompt_type
    # Load model and tokenizer
    tokenizer, query_pipeline = load_model(args.model_path)
    
    # Load dataset
    df = pd.read_csv(args.input_file)
    print(f"Loaded dataset with shape: {df.shape}")
    
    # Generate prompts
    prompts = generate_prompts(df)
    
    # Generate counterspeech responses
    responses = test_model(tokenizer, query_pipeline, prompts)
    
    # Save results
    result_df = pd.DataFrame({"Hatespeech": df['Hatespeech'], "Counterspeech": responses})
    result_df.to_csv(f"results/{prompt_type}/gpt.csv", index=False)
    # print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
