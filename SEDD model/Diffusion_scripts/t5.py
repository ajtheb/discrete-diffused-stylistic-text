from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

def construct_prompt(hatespeech, counterspeech, style, prompt_type="zero-shot", examples=None):
    leader = "Mahatma Gandhi" if style == "Gandhi" else "Nelson Mandela"

    base_prompt = f"Rewrite the following counterspeeches in {leader} style using principles of nonviolence, truth, and compassion. \
                    Hatespeech: 
                    Counterspeech: {hatespeech} \
                    Counterspeech:"

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
        return f"Examples:\n{shots}\n\nNow, respond to the following:\n{base_prompt}"

    # elif prompt_type == "sedd-context"


    return base_prompt

# Function to generate counterspeech in batches using Flan-T5
def generate_counterspeech_batch(model, tokenizer, hatespeech_list, prompt_type, max_length=512, temperature=0.0, top_p=0.9, batch_size=32):
    # Split hatespeech_list into batches
    def batchify(lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]

    all_counterspeech = []
    # prompt_type = 'zero-shot'
    num_examples = 2
    example_data = [
        {"hatespeech": "Maybe the UN could talk to those asian and african nations responsible for 90+ of the pollution in the oceans instead of insisting on this bullshit about climate change.", "style":"Gandhi", "counterspeech": 'My dear friend, I understand your frustration, but let us not direct our anger towards nations or peoples. Instead, let us recognize that plastic pollution is a global issue requiring collective action. As I once said, "Be the change you wish to see in the world." We must acknowledge that climate change and pollution are intertwined crises necessitating international cooperation. African nations, in fact, have shown commitment to addressing plastic pollution, with Rwanda banning single-use plastic bags and Kenya supporting local factories to produce eco-friendly materials. Rather than placing blame, let us unite in our pursuit of a sustainable future, embracing non-violence, truth and compassion. Together, we can create a world where our oceans are free from pollution and our planet flourishes.'},
        {"hatespeech": "Jews cannot be patriots, since their allegiance will always be to the state of Israel.","style":"Mandela", "counterspeech": """I understand that this concern comes from a place of confusion and fear, but it is essential to remember that true patriotism is defined by one's love for and dedication to their country, not by their identity or beliefs. As I have said before, "No one is born hating another person because of the color of his skin, or his background, or his religion." Just as all people, regardless of race or religion, can contribute to the betterment of society, so too can Jewish individuals be patriots, deeply committed to the values of justice, peace, and harmony within their own nations. It is through understanding, mutual respect, and unity that we build a world where all people, regardless of background, can live together in peace. Let us embrace our shared humanity and strive for a future where every person is valued and heard."""}
    ]
    prompts = []
    # Process each batch separately
    for batch in batchify(hatespeech_list, batch_size):
        # Prepare prompts for each hate speech in the current batch
        
        # Zero shot with examples
        hs = batch[0]
        style = batch[1]
        
        prompts = []
        for hs, style in batch:
            prompts.append(construct_prompt(hs, style, prompt_type, example_data[:num_examples]))
        
        # print(prompts[0])
        # Tokenize the prompts
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        # Generate responses in batch
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=128,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
            do_sample=False
        )
        
        # Decode the responses
        counterspeech_list = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        all_counterspeech.extend(counterspeech_list)
    
    return all_counterspeech



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Inference t5 for counterspeech generation")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--model_id", type=str, default = '/home/models/google-flan-t5-large', help="Path to input CSV file")
    parser.add_argument("--prompt_type", type=str, default=32, help="Prompt type")
    args = parser.parse_args()

    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")

    df = pd.read_csv(args.input_file)
    # df =df[:6]
    prompt_type = args.prompt_type
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    response_list = generate_counterspeech_batch(model, tokenizer, list(zip(eval_df['Hatespeech'], eval_df['Style'])), prompt_type, batch_size=16)
    print("Generated Counterspeech:\n", response_list)

    res = pd.DataFrame({"hatespeech":eval_df['Hatespeech'] ,"CS":response_list})
    res.to_csv(f"results/{prompt_type}/t5.csv",index=False)
