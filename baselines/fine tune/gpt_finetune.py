import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the tokenizer and model
model_name = "/home/models/gpt2-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
# Set pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Load CSV dataset
# train_df = pd.read_csv('/home/aswini/third_project_on_soft_prompt_for_HS/data_analysis/base_train_T5.csv')  # Replace with the path to your training CSV file
# validation_df = pd.read_csv('/home/aswini/third_project_on_soft_prompt_for_HS/data_analysis/base_test_T5.csv')  # Replace with the path to your validation CSV file

data_df = pd.read_csv('/home/aswini/Forth_project_on_Person_inspired_CS/baselines/input/Person_specific_counterspeech3.csv')
random_seed = 42
test_size = 0.2
val_size = 0.1
train_df, test_df = train_test_split(data_df, test_size=test_size, random_state=random_seed)
train_df, eval_df = train_test_split(train_df, test_size=val_size, random_state=random_seed)


# Function to format dataset for DialoGPT
def format_data(df):
    conversations = []
    for _, row in df.iterrows():
        conv = f"{row['Hatespeech']} {tokenizer.eos_token} {row['Style']} {tokenizer.eos_token} {row['Counterspeech']} {tokenizer.eos_token}"
        conversations.append(conv)
    return conversations

# Create Dataset objects
train_dataset = Dataset.from_dict({"text": format_data(train_df)})
validation_dataset = Dataset.from_dict({"text": format_data(eval_df)})

dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset
})

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset_dict.map(tokenize_function, batched=True, remove_columns=["text"])

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results_gpt2",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.02,
    logging_dir="./logs/output_logs",
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Masked Language Modeling is not used for causal language modeling
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_GPT2")
tokenizer.save_pretrained("./fine_tuned_GPT2")
