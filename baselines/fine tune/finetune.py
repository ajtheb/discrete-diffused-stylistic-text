import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import os
# import evaluate

from transformers import GenerationConfig
from time import perf_counter
from sklearn.model_selection import train_test_split
import pandas as pd
from trl import SFTTrainer, SFTConfig

# os.environ['http_proxy'] = "http://xen03.iitd.ac.in:3128"
# os.environ['https_proxy'] = "http://xen03.iitd.ac.in:3128"

# bleu_metric = evaluate.load("bleu")

# def compute_metrics(eval_preds):
#     preds, labels = eval_preds

#     # Decode predictions and labels
#     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     # BLEU requires tokenized input
#     tokenized_preds = [pred.split() for pred in decoded_preds]
#     tokenized_labels = [[label.split()] for label in decoded_labels]  # BLEU expects a list of references

#     bleu_score = bleu_metric.compute(predictions=tokenized_preds, references=tokenized_labels)

#     return {
#         "bleu": bleu_score["bleu"],
#     }


def formatted_prompt(question)-> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant:"

def formatted_train(input,response)->str:
    return f"<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n"

def generate_response(user_input):
  prompt = formatted_prompt(user_input)
  inputs = tokenizer([prompt], return_tensors="pt")
  generation_config = GenerationConfig(penalty_alpha=0.6,do_sample = True,
      top_k=5,temperature=0.5,repetition_penalty=1.2,
      max_new_tokens=60,pad_token_id=tokenizer.eos_token_id
  )
  start_time = perf_counter()
  inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
  outputs = model.generate(**inputs, generation_config=generation_config)
  theresponse = (tokenizer.decode(outputs[0], skip_special_tokens=True))
  print(tokenizer.decode(outputs[0], skip_special_tokens=True))
  output_time = perf_counter() - start_time
  print(f"Time taken for inference: {round(output_time,2)} seconds")

def get_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.add_special_tokens({'additional_special_tokens': ["<|im_start|>", "<|im_end|>"]})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        #   quantization_config=bnb_config, 
        device_map={"":0}
    )
    model.config.use_cache=False
    model.config.pretraining_tp=1
    model.resize_token_embeddings(len(tokenizer))
    model.to("cuda")
    return model, tokenizer


def make_prompt(data):
    # print(data)
    if(data['Style']=='Gandhi'):
      leader = 'Mahatma Gandhi'
    else:
       leader = 'Nelson Mandela'
    hs = data['Hatespeech']
    prompt = f"You are {leader}. Respond to the following hate speech with a counterspeech rooted in nonviolence, truth, peace, and understanding.\
               Hate speech: {hs} \
               Counterspeech:"
    return prompt


def prepare_train_datav2(data_df, test_size = 0.2, val_size = 0.1, random_seed = 42):
    
    # make prompts
    data_df['prompt'] = data_df[['Hatespeech', 'Style']].apply(lambda x: make_prompt(x) , axis=1)
    # Create a new column called "text"
    data_df["text"] = data_df[["prompt", "Counterspeech"]].apply(lambda x: "<|im_start|>user\n" + x["prompt"] + " <|im_end|>\n<|im_start|>assistant\n" + x["Counterspeech"] + "<|im_end|>\n", axis=1)
    
    # Train(0.9*0.8), eval(0.1*0.8), test(0.2)
    train_df, test_df = train_test_split(data_df, test_size=test_size, random_state=random_seed)

    train_df, eval_df = train_test_split(train_df, test_size=val_size, random_state=random_seed)

    # Convert to Hugging Face dataset format
    train_data = Dataset.from_pandas(train_df)
    eval_data = Dataset.from_pandas(eval_df)
    test_data = Dataset.from_pandas(test_df)

    return train_data, eval_data, test_data

model_id="/home/models/Mistral-7B-Instruct-v0.2"

model, tokenizer = get_model_and_tokenizer(model_id)

training_data = pd.read_csv('/home/aswini/Forth_project_on_Person_inspired_CS/baselines/input/Person_specific_counterspeech3.csv')

train_data, eval_data, test_data = prepare_train_datav2(training_data)

# print(train_data['text'])
r = 16
peft_config = LoraConfig(
    r=r,  # Increase rank to capture more information (16 → 32)
    lora_alpha=2*r,  # Higher scaling factor for better adaptation (32 → 64)
    lora_dropout=0.1,  # Slightly increase dropout to prevent overfitting (0.05 → 0.1)
    bias="none",
    task_type="CAUSAL_LM"
)

output_model=f"Mistral-7b-Fine-tuned-split-r-{r}"

training_arguments = TrainingArguments(
        output_dir=output_model,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=16,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=8,
        max_steps=250,
        fp16=True,
        push_to_hub=True
    )



trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset = eval_data,
        peft_config=peft_config,
        # args=training_arguments,
        # device= "cuda:1",
        tokenizer=tokenizer,
        args = SFTConfig(
            output_dir=output_model,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=16,
            optim="paged_adamw_32bit",
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            save_strategy="epoch",
            logging_steps=10,
            num_train_epochs=12,
            fp16=True,
            push_to_hub=True,
            packing=False,
            max_seq_length=512,
            dataset_text_field = 'text',
            evaluation_strategy="epoch",
            warmup_ratio = 0.1
        ),
    )

trainer.train()

results = trainer.evaluate()
print(results)