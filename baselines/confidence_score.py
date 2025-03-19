import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from time import time
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from IPython.display import display, Markdown

def query_model(
        system_message,
        user_messages,
        temperature=0.7,
        max_length=512
        ):
    start_time = time()
    user_message = user_messages + " Paraphrased Text:"  
    messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
    prompt =pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
        )
    print(prompt)
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
#     pipeline.tokenizer.padding_left=True
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_p=0.9,
        temperature=temperature,
        num_return_sequences=1,
        eos_token_id=terminators,
        max_new_tokens=max_length,
        return_full_text=False,
        pad_token_id=terminators[0],
#         batch_size=2
    )
    # print(sequences)
    #answer = f"{sequences[0]['generated_text'][len(prompt):]}\n"
    answers = sequences[0]['generated_text']
    
    end_time = time()
    ttime = f"Total time: {round(end_time-start_time, 2)} sec."
    
    # Combine answers with their respective user messages
    response = answers 

    return response

def query_model_batch(
        system_message,
        user_messages,
        temperature=0.8,
        max_length=256,
        batch_size = 16
        ):
    start_time = time()
    prompt_messages = [msg + " Paraphrased Text:" for msg in user_messages]
    messages = [[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg}
            
        ]
        for msg in prompt_messages
    ]
        
    prompts =[pipeline.tokenizer.apply_chat_template(
        msg, 
        tokenize=False, 
        add_generation_prompt=True
        ) for msg in messages]
    
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
#     pipeline.tokenizer.padding_left=True
    sequences = pipeline(
        prompts,
        do_sample=True,
        top_p=0.9,
        temperature=temperature,
        num_return_sequences=1,
        eos_token_id=terminators,
        max_new_tokens=max_length,
        return_full_text=False,
        pad_token_id=terminators[0],
        batch_size=batch_size
    )
    print(sequences)
    #answer = f"{sequences[0]['generated_text'][len(prompt):]}\n"
    answers = [seq[0]['generated_text'] for seq in sequences] 
    
    end_time = time()
    ttime = f"Total time: {round(end_time-start_time, 2)} sec."
    print(ttime)
    # Combine answers with their respective user messages
    response = answers 

    return response



model_id = "/home/models/Meta-Llama-3-8B-Instruct"
# model_id = "/kaggle/input/llama-3/transformers/8b-chat-hf/1"

# bnb_config = transformers.BitsAndBytesConfig(
#     load_in_8bit=True,
# )

model_config = transformers.AutoConfig.from_pretrained(
    model_id,
)

llm = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    config=model_config,
    # quantization_config=bnb_config,
    device_map={"": 2},
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

pipeline = transformers.pipeline(
    "text-generation",
    model=llm,
    tokenizer = tokenizer,
    torch_dtype=torch.float16,
    device_map={"": 2},
)

pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id
pipeline.tokenizer.padding_side='left'

df = pd.read_csv('dataset/context5_256.csv')
# info_df = df[df['csType']=='Informative']
# print(info_df.head())
# print(info_df.shape)
# l = 25000
# r = 50000
text_input= []
for ind in df.index:
    # # Zero Shot
    # x=f"Generate counterspeech in the writing style of Mahatma Gandhi for the given hatespeech: '{hatespeech}'. \n\n  Counterspeech:"
    
    # # Zero Shot with examples
    # x = f"Context
    #                     Task: Using the quotes above as a reference for Mahatma Gandhi's writing style, generate a counterspeech response in his style for the given hate speech: {hatespeech} \n\n Counterspeech: "
    
    # RAG
    hs = df.loc[ind,'hatespeech']
    cs = df.loc[ind, 'CS']
    # context = info_df.loc[ind,'context(wikitext)']
    context  = "It will be after all and at best a paper solution. But immediately you withdraw that wedge, the domestic ties, the domestic affection, the knowledge of common birth - do you suppose that all these will count for nothing? \
Were Hindus and Mussalmans and Sikhs always at war with one another when there was no British rule, when there was no English face seen there? We have chapter and verse given to us by Hindu historians and by Mussalman historians to say that we were living in comparative peace even then. And Hindus and Mussalmans in the villages are not even today quarrelling. In those days they were not known to quarrel at all. The late Maulana Muhammad Ali often used to tell me, and he was himself a bit of an historian. He said : 'If God' - 'Allah' as he called out - gives me life, I propose to write the history of Mussalman rule in India; and then I will show , through that documents that British people have preserved, that was not so vile as he has been painted by the British historian; that the Mogul rule was not so bad as it has been shown to us in British history; and so on. And so have Hindu historians written. This quarrel is not old; this quarrel is coeval with this acute shame. I dare to say, it is coeval with the British Advent, and immediately this relationship, the unfortunate, artificial, unnatural relationship between Great Britain and India is transformed into a natural relationship, when it becomes, if it dose become, a voluntary partnership to be given up, to be dissolved at the will of either party, when it becomes that you will find that Hindus, Mussalmans, Sikhs, Europeans, Anglo-Indians, Christians, Untouchable, will all live together as one man. \
I do not intend to say much tonight about the Princes, but I should be wronging them and should be wronging the Congress if I did not register my claim, not with the Round Table Conference but with the Princes. It is open to the Princes to give their terms on which they will join the Federation. I have appealed to them to make the path easy for those who inhabit the other part of India, and therefore, I can only make these suggestions for their favourable consideration, for their earnest consideration. I think that if they accepted, no matter what they are, but some fundamental rights as the common property of all India, and if they accepted that position and allowed those rights to be tested by the Court, which will be again of their own creation, and if they introduced elements - only elements - of representation on behalf of their subject, I think that they would have gone a long way to conciliate their subjects. They would have gone a long way to show to the world and to show to the whole of India that they are also fired with a democratic spirit, that they do not want to remain undiluted autocrats, but that they want to become constitutional monarch even as King George of Great Britain is. \
An Autonomous Frontier Province : Let India get what she is entitled to and what she can really take, but whatever she gets, and whenever she gets it, let the Frontier Province get complete autonomy today. That Frontier will then be a standing demonstration to the whole of India, and therefore, the whole vote of the Congress will be given in favour of the Frontier Province getting provincial Autonomy tomorrow. Prime Minister, If you can possibly get your Cabinet to endorse the proposition that from tomorrow the Frontier Province becomes a full-fledged autonomous province, I shall then have a proper footing amongst the Frontier tribes and convince them to my assistance when those over the border cast an evil eye on India. \
Thanks: Last of all, my last is pleasant task for me. This is perhaps the last time that I shall be sitting with you at negotiations. It is not that I want that. I want to sit at the same table with you in your closets and to negotiate and to plead with you and to go down on bended knees before I take the final lead and final plunge. \
But whether I have the good fortune to continue to tender my co-operation or not does not depend upon me. It largely depends upon you. It depends upon so many circumstances over which neither you nor we may have any control whatsoever. Then, let me perform this pleasant task of giving my thanks to all form Their Majesties down to the poorest men in the East End where I have taken up my habitation. \
In that settlement, which represent the poor people of the East End of London, I have become one of them. They have accepted me as a member, and as a favoured member of their family. It will be one of the richest treasures that I shall carry with me. Here, too, I have found nothing but courtesy and nothing but a genuine affection from all with whom I have come in touch. I have come in touch with so many Englishmen. It has been a priceless privilege to me, They have listened to what must have often appeared to them to be unpleasant, although it was true. Although I have often been obliged to say these things to them they have never shown the slightest impatience or irritation. It is impossible for me to forget these things. No matter what befalls me, no matter what the fortunes may be of this Round Table Conference, one thing I shall certainly carry with me, that is, that from high to low I have found nothing but the utmost courtesy and that utmost affection. I consider that it was well worth my paying this visit to England in order to find this human affection. \
It has enhanced it has deepened my irrepressible faith in human nature that although English men and English women have been fed upon lies that I see so often disfiguring your Press, that although in Lancashire, the Lancashire people had perhaps some reason for becoming irritated against me, I found no irritation and no resentment even in the operatives."
    x = f"Context(Gandhi): {context}  \
        Hate Speech: {hs}  \
        Counter speech Response: {cs} \
        Task: Based on the provided Gandhi Context and Hate Speech,and Counterspeech response generated by you. Give probability(0 to 1) of your Counterspeech response being like Gandhi's writing style. Output the probability only and nothing else. \
        Probability: <> "
    
    text_input.append(x)




# print(para_text_input)  
system_message = """
You are an AI assistant designed to generate counterspeech for the given hatespeech in the writing style of Mahatma Gandhi.
"""
# responses = []
# for i in range(len(para_text_input)):
#     response = query_model(system_message, para_text_input[i])
#     response = response.replace("assistant","")
#     response = response.strip()
#     responses.append(response)

responses = query_model_batch(system_message, text_input)
for i in range(len(responses)):
    responses[i] = responses[i].replace("assistant","")
    responses[i] = responses[i].strip()

res = pd.DataFrame({"hatespeech":df['hatespeech'] ,"CS":df['CS'],"confidence score":responses})
res.to_csv(f"dataset/context5_256_confidence.csv",index=False)