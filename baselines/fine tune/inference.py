import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig,BitsAndBytesConfig
from peft import PeftModel
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
import transformers

# Set model path
fine_tuned_model_path = "Mistral-7b-Fine-tuned-split-r-16"

# Load base model and tokenizer
# base_model_id = "/home/models/Meta-Llama-3.1-8B-Instruct"  # Change if needed
base_model_id = "/home/models/Mistral-7B-Instruct-v0.2"  # Change if needed
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.add_special_tokens({'additional_special_tokens': ["<|im_start|>", "<|im_end|>"]})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
# Load fine-tuned model with LoRA
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype="float16", 
        bnb_4bit_use_double_quant=True
    )
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto", torch_dtype=torch.float16, 
                                                #   quantization_config=bnb_config
                                                  )
model = PeftModel.from_pretrained(base_model, fine_tuned_model_path)
model.to("cuda")
model.resize_token_embeddings(len(tokenizer))

def make_prompt(data):
    # print(data)
    if(data['Style']=='Gandhi'):
      leader = 'Mahatma Gandhi'
    else:
       leader = 'Nelson Mandela'
    hs = data['Hatespeech']
    prompt = f"You are {leader}. Respond to the following hate speech with a counterspeech rooted in nonviolence, truth, peace, and understanding of {leader}.\
               Hate speech: {hs} \
               Counterspeech: "
    return prompt

def prepare_train_datav2(data_df, test_size = 0.2, val_size = 0.1, random_seed = 42):
    
    # make prompts
    data_df['prompt'] = data_df[['Hatespeech', 'Style']].apply(lambda x: make_prompt(x) , axis=1)
    # Create a new column called "text"
    data_df["text"] = data_df[["prompt", "Counterspeech"]].apply(lambda x: "<|im_start|>user\n" + x["prompt"] + " <|im_end|>\n<|im_start|>assistant\n", axis=1)
    
    # Train(0.9*0.8), eval(0.1*0.8), test(0.2)
    train_df, test_df = train_test_split(data_df, test_size=test_size, random_state=random_seed)

    train_df, eval_df = train_test_split(train_df, test_size=val_size, random_state=random_seed)

    # Convert to Hugging Face dataset format
    train_data = Dataset.from_pandas(train_df)
    eval_data = Dataset.from_pandas(eval_df)
    test_data = Dataset.from_pandas(test_df)

    return train_data, eval_data, test_data

def generate_response(user_input):
    # Format the prompt
    # prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant:"
    prompt = user_input
    print("---")
    print(prompt)
    print("---")
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt",padding=True, truncation=True).to("cuda")

    end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    # Define generation configuration
    generation_config = GenerationConfig(
        do_sample=False, 
        top_k=5, 
        temperature=0.5, 
        repetition_penalty=1.2,
        max_new_tokens=240, 
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id = end_token_id
    )

    # Generate response
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # print(response)
    # print("---")
    # Extract only the assistant's response
    response = response.split("<|im_start|>assistant")[-1].strip()
    
    return response

def query_model_batch(pipeline, system_message, user_messages, temperature=0.0, max_length=256, batch_size=16):
    # start_time = time()
    
    prompts = user_messages
    
    terminators = [
        pipeline.tokenizer.convert_tokens_to_ids("<|im_end|>")
    ]
    
    sequences = pipeline(
        prompts,
        do_sample=False,
        num_return_sequences=1,
        eos_token_id=terminators[0],
        max_new_tokens=max_length,
        return_full_text=False,
        pad_token_id=pipeline.tokenizer.eos_token_id,
        batch_size=batch_size
    )
    
    responses = [seq[0]['generated_text'] for seq in sequences]
    
    # print(f"Total time: {round(time() - start_time, 2)} sec.")
    return responses


pipe = transformers.pipeline(task="text-generation", model=model, tokenizer=tokenizer)


training_data = pd.read_csv('/home/aswini/Forth_project_on_Person_inspired_CS/baselines/input/Person_specific_counterspeech3.csv')

train_data, eval_data, test_data = prepare_train_datav2(training_data)


# eval_data = eval_data[:10]
eval_texts = test_data["text"]  # Extracting evaluation set texts
hatespeech = test_data["Hatespeech"]
generated_responses = []
# eval_texts = eval_texts[:20]
# print(eval_texts)
# print(type(eval_texts))
responses = query_model_batch(pipe, "", eval_texts)
# print(responses)v
# print(generated_responses)

# clean_res = []
# for response in responses:
#     match = re.search(r"<\|im_start\|>assistant\s*(.*?)<\|im_end\|>", response, re.DOTALL)
#     if match:
#         clean_res.append(match.group(1).strip())
#     else:
#         clean_res.append("Miss")

# # Store results in a DataFrame for analysis
eval_results = pd.DataFrame({"prompt": eval_texts, "hatespeech": hatespeech, "CS": responses})

# # Save the results to a CSV file
# eval_results.to_csv("/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/finetune/llama.csv", index=False)
eval_results.to_csv("/home/aswini/Forth_project_on_Person_inspired_CS/baselines/inference_scripts/results/finetune/mistral_r_16.csv", index=False)
