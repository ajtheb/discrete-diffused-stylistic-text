import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from time import time
from IPython.display import display, Markdown
import argparse

# Function to construct prompt
def construct_prompt(hatespeech, style, prompt_type="zero-shot", examples=None):
    leader = "Mahatma Gandhi" if style == "Gandhi" else "Nelson Mandela"

    base_prompt = f"You are {leader}. Respond to the following hate speech with a counterspeech rooted in nonviolence, truth, peace, and understanding. \nHate speech: {hatespeech}\nCounterspeech:"

    if prompt_type == "zero-shot":
        return base_prompt

    elif prompt_type == "one-shot" and examples:
        example = examples[0]
        leader_example = "Mahatma Gandhi" if example["style"] == "Gandhi" else "Nelson Mandela"
        return f"Example:\nHate speech: {example['hatespeech']}\n{leader_example}'s Counterspeech: {example['counterspeech']}\n\nNow, respond to the following:\n{base_prompt}"

    elif prompt_type == "few-shot" and examples:
        shots = "\n\n".join([
            f"Hate speech: {ex['hatespeech']}\n{('Mahatma Gandhi' if ex['style'] == 'Gandhi' else 'Nelson Mandela')}'s Counterspeech: {ex['counterspeech']}"
            for ex in examples
        ])
        return f"Examples:\n{shots}\n\nNow, respond to the following with only counterspeech and nothing else. \n{base_prompt}"

    return base_prompt

# Function to generate responses
def query_model_batch(pipeline, system_message, user_messages, temperature=0.0, max_length=256, batch_size=16):
    start_time = time()
    print(user_messages)
    prompts = [
        pipeline.tokenizer.apply_chat_template(
            [{"role": "user", "content": system_message + msg},
            {"role": "assistant", "content": ""}],
            tokenize=False, add_generation_prompt=True
        ) for msg in user_messages
    ]
    
    
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    sequences = pipeline(
        prompts,
        do_sample=False,
        top_p=0.9,
        temperature=temperature,
        num_return_sequences=1,
        eos_token_id=terminators,
        max_new_tokens=max_length,
        return_full_text=False,
        pad_token_id=terminators[0],
        batch_size=batch_size
    )
    
    responses = [seq[0]['generated_text'] for seq in sequences]
    
    print(f"Total time: {round(time() - start_time, 2)} sec.")
    return responses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference mistral 2 for counterspeech generation")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--model_id", type=str, default = "/home/models/Mistral-7B-Instruct-v0.2", help="Path to input CSV file")
    parser.add_argument("--prompt_type", type=str, default=32, help="Prompt type")
    args = parser.parse_args()
    
    # Initialize the model
    model_id = args.model_id
    model_config = transformers.AutoConfig.from_pretrained(model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
    )

    llm = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        config=model_config,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config = bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = llm.config.eos_token_id
    tokenizer.padding_side = 'left'
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=llm,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
    pipeline.tokenizer.padding_side = 'left'
    print(f"PAD TOKEN: {tokenizer.pad_token}, PAD TOKEN ID: {tokenizer.pad_token_id}")
    print(f"EOS TOKEN: {tokenizer.eos_token}, EOS TOKEN ID: {tokenizer.eos_token_id}")
    print(llm.config.eos_token_id)
    
    # Load dataset
    df = pd.read_csv(args.input_file)
    # df = df[:2]
    example_data = [
        {"hatespeech": "Maybe the UN could talk to those asian and african nations responsible for 90+ of the pollution in the oceans instead of insisting on this bullshit about climate change.", "style":"Gandhi", "counterspeech": 'My dear friend, I understand your frustration, but let us not direct our anger towards nations or peoples. Instead, let us recognize that plastic pollution is a global issue requiring collective action. As I once said, "Be the change you wish to see in the world." We must acknowledge that climate change and pollution are intertwined crises necessitating international cooperation. African nations, in fact, have shown commitment to addressing plastic pollution, with Rwanda banning single-use plastic bags and Kenya supporting local factories to produce eco-friendly materials. Rather than placing blame, let us unite in our pursuit of a sustainable future, embracing non-violence, truth and compassion. Together, we can create a world where our oceans are free from pollution and our planet flourishes.'},
        {"hatespeech": "Jews cannot be patriots, since their allegiance will always be to the state of Israel.","style":"Mandela", "counterspeech": "I understand that this concern comes from a place of confusion and fear, but it is essential to remember that true patriotism is defined by one's love for and dedication to their country, not by their identity or beliefs. As I have said before, \"No one is born hating another person because of the color of his skin, or his background, or his religion.\" Just as all people, regardless of race or religion, can contribute to the betterment of society, so too can Jewish individuals be patriots, deeply committed to the values of justice, peace, and harmony within their own nations. It is through understanding, mutual respect, and unity that we build a world where all people, regardless of background, can live together in peace. Let us embrace our shared humanity and strive for a future where every person is valued and heard."}
    ]
    
    # Choose prompt type: "zero-shot", "one-shot", or "few-shot"
    prompt_type = args.prompt_type
    num_examples = 1  # Number of examples for few-shot learning
    
    # Generate prompts
    text_input = [construct_prompt(df.loc[ind, 'Hatespeech'], df.loc[ind, 'Style'], prompt_type, example_data[:num_examples]) for ind in df.index]
    
    # Run model
    system_message = ""
    responses = query_model_batch(pipeline, system_message, text_input)
    
    # Clean responses
    responses = [resp.replace("assistant", "").strip().rsplit(".", 1)[0] + "." for resp in responses]
    
    # Save results
    res = pd.DataFrame({"hatespeech": df['Hatespeech'], "CS": responses})
    res.to_csv(f"results/{prompt_type}/mistral.csv", index=False)
    
    print("Processing complete! Results saved.")
