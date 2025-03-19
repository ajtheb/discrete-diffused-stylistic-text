import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from sklearn.model_selection import train_test_split
# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the fine-tuned model and tokenizer
model_path = "/home/aswini/Forth_project_on_Person_inspired_CS/baselines/fine tune/results_gpt2/checkpoint-2000"  # Path to your fine-tuned model

if not os.path.exists(model_path):
    raise ValueError(f"The checkpoint path {model_path} does not exist.")
tokenizer_path = "/home/models/gpt2-large"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

# Set pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Load validation CSV dataset
data_df = pd.read_csv('/home/aswini/Forth_project_on_Person_inspired_CS/baselines/input/Person_specific_counterspeech3.csv')
random_seed = 42
test_size = 0.2
val_size = 0.1
train_df, test_df = train_test_split(data_df, test_size=test_size, random_state=random_seed)
train_df, eval_df = train_test_split(train_df, test_size=val_size, random_state=random_seed)

# Function to generate responses
def generate_response(prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,  # Enable sampling for more diverse responses
        # top_k=50,  # Use top-k sampling
        # top_p=0.95,  # Use top-p sampling
        # temperature=0.9  # Adjust temperature for sampling
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Generate responses for validation data
responses = []
for index, row in test_df.iterrows():
    prompt = f"{row['Hatespeech']} {tokenizer.eos_token} {row['Style']} {tokenizer.eos_token} {tokenizer.eos_token}"
    response = generate_response(prompt)
    responses.append(response)
    # print("Hate Speech:")
    # print("--------------------------------")
    # print(row['Hatespeech'])
    # print("--------------------------------")
    # print("Generated Counter speech")
    # print("-------------------------------")
    # print(response)
    # print("--------------------------------")

# Add responses to the validation dataframe
test_df['CS'] = responses

# Save the validation dataframe with responses to a new CSV file
# test_df.to_csv('', index=False)
test_df.to_csv("/home/aswini/Forth_project_on_Person_inspired_CS/baselines/inference_scripts/results/finetune/gpt2.csv",index=False)


print("Inference completed and responses saved")
