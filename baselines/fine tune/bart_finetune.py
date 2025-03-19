import argparse
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_train_datav2(data_df, test_size=0.2, random_seed=42):
    train_df, eval_df = train_test_split(data_df, test_size=test_size, random_state=random_seed)
    train_data = Dataset.from_pandas(train_df)
    eval_data = Dataset.from_pandas(eval_df)
    return train_data, eval_data

def preprocess_function(examples):
    inputs = [f"Generate a counterspeech in {style}'s style: {text}" for text, style in zip(examples['Hatespeech'], examples['Style'])]
    targets = [response for response in examples['Counterspeech']]
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=512, truncation=True, padding='max_length')
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

def main(args):
    global tokenizer, model
    
    tokenizer = BartTokenizer.from_pretrained(args.model_path)
    model = BartForConditionalGeneration.from_pretrained(args.model_path)
    
    training_data = pd.read_csv(args.data_path)
    train_data, eval_data = prepare_train_datav2(training_data, test_size=args.test_size, random_seed=args.seed)
    
    tokenized_dataset_train = train_data.map(preprocess_function, batched=True)
    tokenized_dataset_eval = eval_data.map(preprocess_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_eval,
    )
    
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/models/google-flan-t5-large", help="Path to the model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset CSV file")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save model outputs")
    parser.add_argument("--learning_rate", type=float, default=4e-5, help="Learning rate for training")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test dataset split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    main(args)
