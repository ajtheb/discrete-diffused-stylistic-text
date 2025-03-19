import transformers
import torch
import pandas as pd
import argparse
from time import time
from tqdm import tqdm

def load_model(model_id):
    """Loads the BART model and tokenizer."""
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    
    pipeline = transformers.pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    
    return tokenizer, pipeline

def get_prompt(instruction):
    """Formats the instruction prompt."""
    return f"{instruction}"

def generate_prompts(df, prompt_type='zero-shot', examples=None):
    """Generates prompts for counterspeech generation."""
    prompts = []
    for ind in tqdm(df.index, desc="Processing rows"):
        hatespeech = df.loc[ind, 'Hatespeech']
        style = df.loc[ind, 'Style']
        leader = 'Mahatma Gandhi' if style == 'Gandhi' else 'Nelson Mandela'
        
        prompt = (f"You are {leader}. Respond to the following hate speech with a counterspeech rooted in "
                  f"nonviolence, truth, peace, and understanding. Use one of your quotes if appropriate.\n\n"
                  f"Hate speech: {hatespeech}\n\nCounterspeech:")
        
        if prompt_type=='few-shot':
           shots = "\n\n".join([
                                f"Hate speech: {ex['hatespeech']}\n{('Mahatma Gandhi' if ex['style'] == 'Gandhi' else 'Nelson Mandela')}'s Counterspeech: {ex['counterspeech']}"
                                for ex in examples
                                ])
           prompt = f"Examples:\n{shots}\n\nNow, respond to the following:\n{prompt}"
        
        prompts.append(get_prompt(prompt))
    return prompts

# def construct_prompt(hatespeech, style, prompt_type="zero-shot", examples=None):
#     leader = "Mahatma Gandhi" if style == "Gandhi" else "Nelson Mandela"

#     base_prompt = f"You are {leader}. Generate a counterspeech response to the following hate speech, rooted in nonviolence, truth, peace, and understanding.  \
#                     If relevant, include one of your quotes to strengthen your response. \
#                     Hate speech: {hatespeech} \
#                     Counterspeech:"

#     if prompt_type == "zero-shot":
#         return base_prompt

#     elif prompt_type == "one-shot" and examples:
#         example = examples[0]
#         leader_example = "Mahatma Gandhi" if example["style"] == "Gandhi" else "Nelson Mandela"
#         return f"Example:\nHate speech: {example['hatespeech']}\n{leader_example}'s Counterspeech: {example['counterspeech']}\n\nNow, respond to the following:\n{base_prompt}"

#     elif prompt_type == "few-shot" and examples:
#         shots = "\n\n".join([
#             f"Hate speech: {ex['hatespeech']}\n{('Mahatma Gandhi' if ex['style'] == 'Gandhi' else 'Nelson Mandela')}'s Counterspeech: {ex['counterspeech']}"
#             for ex in examples
#         ])
#         return f"Examples:\n{shots}\n\nNow, respond to the following:\n{base_prompt}"

#     return base_prompt

def test_model(tokenizer, pipeline, prompts, batch_size=32):
    """Runs the model on input prompts and returns generated responses."""
    start_time = time()
    sequences = pipeline(
        prompts,
        do_sample=False,
        top_k=10,
        num_return_sequences=1,
        max_new_tokens=512,
        batch_size=batch_size
    )
    elapsed_time = round(time() - start_time, 3)
    print(f"Test inference time: {elapsed_time} sec.")
    
    return [seq['generated_text'].strip() for seq in sequences]

def main():
    parser = argparse.ArgumentParser(description="Generate counterspeech using a BART model")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--prompt_type", type=str, required=True, help="Prompt type")
    parser.add_argument("--model_path", type=str, default="/home/models/bart-large", help="Path to the model")
    args = parser.parse_args()
    
    prompt_type = args.prompt_type
    # Load model and tokenizer
    tokenizer, query_pipeline = load_model(args.model_path)
    
    # Load dataset
    df = pd.read_csv(args.input_file)
    print(f"Loaded dataset with shape: {df.shape}")
    
    
    example_data = [
        {"hatespeech": "Maybe the UN could talk to those asian and african nations responsible for 90+ of the pollution in the oceans instead of insisting on this bullshit about climate change.", "style":"Gandhi", "counterspeech": 'My dear friend, I understand your frustration, but let us not direct our anger towards nations or peoples. Instead, let us recognize that plastic pollution is a global issue requiring collective action. As I once said, "Be the change you wish to see in the world." We must acknowledge that climate change and pollution are intertwined crises necessitating international cooperation. African nations, in fact, have shown commitment to addressing plastic pollution, with Rwanda banning single-use plastic bags and Kenya supporting local factories to produce eco-friendly materials. Rather than placing blame, let us unite in our pursuit of a sustainable future, embracing non-violence, truth and compassion. Together, we can create a world where our oceans are free from pollution and our planet flourishes.'},
        {"hatespeech": "Jews cannot be patriots, since their allegiance will always be to the state of Israel.","style":"Mandela", "counterspeech": """I understand that this concern comes from a place of confusion and fear, but it is essential to remember that true patriotism is defined by one's love for and dedication to their country, not by their identity or beliefs. As I have said before, "No one is born hating another person because of the color of his skin, or his background, or his religion." Just as all people, regardless of race or religion, can contribute to the betterment of society, so too can Jewish individuals be patriots, deeply committed to the values of justice, peace, and harmony within their own nations. It is through understanding, mutual respect, and unity that we build a world where all people, regardless of background, can live together in peace. Let us embrace our shared humanity and strive for a future where every person is valued and heard."""}
    ]

    num_examples = 2
    # Generate prompts
    prompts = generate_prompts(df, example_data[:num_examples])
    
    # Generate counterspeech responses
    responses = test_model(tokenizer, query_pipeline, prompts)
    
    # Save results
    result_df = pd.DataFrame({"Hatespeech": df['Hatespeech'], "Counterspeech": responses})
    result_df.to_csv(f"results/{prompt_type}/bart.csv", index=False)
    # print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
