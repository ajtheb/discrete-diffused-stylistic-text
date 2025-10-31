import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from peft import IA3Config, get_peft_model
from datasets import Dataset
import pandas as pd

# Sample few-shot dataset (replace with real counterspeech dataset)
# data = {
#     "text": [
#         "You're worthless and nobody cares about you.",
#         "Immigrants are ruining this country.",
#         "Women don't belong in tech.",
#         "You're a disgrace to society.",
#         "Gays shouldn't be allowed in public spaces."
#     ],
#     "label": [1, 1, 1, 1, 1]  # 1 for counterspeech needed, adjust for task
# }
df = pd.read_csv('Diffusion_scripts/Train.csv')
gandhi_data = df[df['Style'] == 'Gandhi'].sample(25, random_state=42)
mandela_data = df[df['Style'] == 'Mandela'].sample(25, random_state=42)

# Combine hate speech, style, and counterspeech in text column
data = {
    'text': (
        [f"Hate speech: {row['Hatespeech']}. Style: Gandhi. Counterspeech: {row['Counterspeech']}." for _, row in gandhi_data.iterrows()] +
        [f"Hate speech: {row['Hatespeech']}. Style: Mandela. Counterspeech: {row['Counterspeech']}." for _, row in mandela_data.iterrows()]
    ),
    'label': [0] * 25 + [1] * 25  # 0: Gandhi, 1: Mandela
}
df_fs = pd.DataFrame(data)
dataset = Dataset.from_pandas(df_fs)

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Tokenize dataset
def preprocess_function(examples):
    # Tokenize the text and include labels
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    tokenized["labels"] = examples["label"]
    return tokenized

# Apply preprocessing to the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Remove unnecessary columns and set format for PyTorch
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])

# Add (IA)³ configuration
ia3_config = IA3Config(
    target_modules=["key", "value", "intermediate.dense", "output_layer_norm"],
    feedforward_modules=["intermediate.dense"],
    task_type="SEQ_CLS"
)

# Apply (IA)³ to the model
model = get_peft_model(model, ia3_config)

# Print number of trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params} ({trainable_params/total_params*100:.4f}% of total)")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./ia3_bert_counterspeech",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",
    metric_for_best_model="accuracy",
    logging_first_step=True,
    seed=42,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./ia3_bert_counterspeech")
tokenizer.save_pretrained("./ia3_bert_counterspeech")

# Save (IA)³ weights separately (optional)
torch.save(model.state_dict(), "./ia3_bert_counterspeech.pth")