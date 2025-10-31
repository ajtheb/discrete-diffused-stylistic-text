import os
import sys
sys.path.append('/home/aswini/Forth_project_on_Person_inspired_CS/Diffusion_Language_Modeling/')
import torch
import argparse
import yaml
from transformers import BertTokenizer, BertModel
from transformers import GPT2TokenizerFast

from sedd.models.sedd_BERT2 import SEDD_BERT
from sedd.models.sedd_BERT3 import SEDD_BERT_CLIME, SEDD_BERT_COGENT
from sedd.models.sedd import SEDD
from sedd.models.graph import AbsorbingGraph
from sedd.models.noise import LogLinearNoise
from sedd.models.sampler import Sampler
import pandas as pd
from tqdm import tqdm
import time
from sklearn.preprocessing import LabelEncoder



def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Generate samples in batch")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_type", default="normal", type=str)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--show_intermediate", action='store_true')
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--vector_quantization", type=int,default=0, required=True, help="Flag for vector quantization(0/1)")
    parser.add_argument("--save_dir", type=str, required=True, help="save_path")
    parser.add_argument("--clime_model", type=str,default="", help="SEDD_BERT_CLIME")
    parser.add_argument("--checkpoint_file", type=str,default="", help="specify particular checkpoint")
    parser.add_argument("--input_csv", type=str,default="Diffusion_scripts/input/Test_new.csv", help="Input data file")
    
    args = parser.parse_args()
    df_train = pd.read_csv("Diffusion_scripts/input/Train_new.csv")
    df = pd.read_csv(args.input_csv)

    # Initialize LabelEncoder
    le = LabelEncoder()

    # Fit and transform on training data
    df_train['Target'] = le.fit_transform(df_train['Target'])

    # Transform evaluation data using the same encoder
    df['Target'] = le.transform(df['Target'])

    # df_train['Style'] = df_train['Style'].map({"Gandhi":0, "Mandela":1})
    df['Style'] = df['Style'].map({"Gandhi":0, "Mandela":1})
    
    print(df['Target'].describe())
    # Add mapping print here
    print("\nLabel Encoder Mappings:")
    for i, class_name in enumerate(le.classes_):
        print(f"{class_name} → {i}")
    
    # df = df[:64]
    # Record start time
    
    
    total_generated_tokens = 0

    # Config should be saved in the model directory
    cfg = os.path.join(args.model, 'config.yaml')
    with open(cfg, 'r') as f:
        cfg = yaml.full_load(f)
    vq_flag = args.vector_quantization
    if(vq_flag):
        cfg['model']['use_style_embedding'] = True
    else:
        cfg['model']['use_style_embedding'] = False
        
    model_type = args.model_type
    # Load the tokenizer
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if(model_type=='normal'):
        # Initialize tokenizer
        tokenizer = GPT2TokenizerFast.from_pretrained('/home/models/gpt2')
        tokenizer.pad_token = '<PAD>'
        # for gpt2
        model = SEDD(cfg, tokenizer.vocab_size).to(args.device)
        cfg['model']['style_dim'] = 50258
        
    elif(model_type=='bert'):
        # for bert
        tokenizer = BertTokenizer.from_pretrained('/home/models/bert-base-uncased') 
        model = SEDD_BERT(cfg).to(args.device)
        cfg['model']['style_dim'] = 1024
    
    elif(model_type=='bert_clime'):
        # for bert
        tokenizer = BertTokenizer.from_pretrained('/home/models/bert-base-uncased') 
        model = SEDD_BERT_CLIME(cfg).to(args.device)
    
    elif(model_type=='bert_cogent'):
        clime_model = SEDD_BERT_CLIME(cfg)
        for param in clime_model.parameters():
            param.requires_grad_(False)
        # Load trained codebooks 
        model_file = os.path.join(args.clime_model, "checkpoint.pth")
        loaded_state = torch.load(model_file, map_location='cuda')
        
        clime_model.load_state_dict(loaded_state, strict=False)
        clime_model.eval()
        # for bert
        tokenizer = BertTokenizer.from_pretrained('/home/models/bert-base-uncased') 
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
        model = SEDD_BERT_COGENT(cfg, clime_model).to(args.device)
        
    # tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    # tokenizer.pad_token = '<PAD>'
    
    print("Vocab size: ", tokenizer.vocab_size)
    print("Last token in vocab: ", tokenizer.decode([tokenizer.vocab_size-1]))
    print("Past vocab size added in the vocabulary EMPTY_STRING: ", tokenizer.batch_decode([[tokenizer.vocab_size]]))

    print("Complete prefix: ", args.prefix)
    print("with suffix: ", args.suffix)

    

    
    # Load the model onto GPU
    #device = torch.device('cuda')
    # model = SEDD(cfg, tokenizer.vocab_size).to(args.device)
    # model = SEDD_BERT(cfg).to(args.device)
    if(args.checkpoint_file==""):
        model_file = os.path.join(args.model, "checkpoint.pth")
    else:
        model_file = os.path.join(args.model, args.checkpoint_file)
    
    loaded_state = torch.load(model_file, map_location=args.device)
    # filtered_state = {
    #     k: v for k, v in loaded_state.items() 
    #     if k in model.state_dict() and not k.startswith('style_quantizer.')
    # }
    model.load_state_dict(loaded_state, strict=False)
    
    # print(model.style_quantizer.embedding.weight)
    # model.load_state_dict(loaded_state)

    # Load the transition graph
    graph = AbsorbingGraph(tokenizer.vocab_size)

    #print(type(graph))
    
    # Load the noise function
    noise = LogLinearNoise().to(args.device)

    
    
    counterspeech_list = []
    df['CS'] = ""
    batch_size = 64  # Adjust based on GPU memory
    num_full_batches = len(df) // batch_size
    
    # Record sampling start time
    sampling_start_time = time.time()
    # df =df[:128]
    for batch_start in tqdm(range(0, num_full_batches * batch_size, batch_size)):
        batch = df.iloc[batch_start:batch_start+batch_size]
        
        # Dynamic input construction based on model type
        batch_inputs = []
        for row in batch.itertuples():
            
            if(row.Style==0):
                style_text = 'Gandhi'
            else:
                style_text = 'Mandela'
                
            
            target_text = le.inverse_transform([row.Target])
            if model_type == 'bert_clime':
                input_text = f"[CLS] {row.Counterspeech} [SEP]"
            else:
                # Original format for other models
                input_text = f"[CLS] {row.Hatespeech} [SEP] {style_text} [SEP] {target_text[0]} [SEP] "
            batch_inputs.append(input_text)
        prefix_encodings = tokenizer(
            batch_inputs,
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt',
            add_special_tokens=False  # Special tokens already added manually
        )
        
        input_ids = prefix_encodings.input_ids.to(args.device)
        attention_mask = prefix_encodings.attention_mask.to(args.device)
        prefix_lens = attention_mask.sum(dim=1)

        def batch_proj_fun(x):
            #  Create indices tensor [0, 1, ..., seq_len-1] for all batch items
            seq_len = x.size(1)
            indices = torch.arange(seq_len, device=x.device).expand(x.size(0), -1)  # [batch_size, seq_len]
            
            # Create boolean mask where indices < prefix lengths
            mask = indices < prefix_lens.unsqueeze(1)  # [batch_size, seq_len]
            
            # Vectorized selection: input_ids where mask=True, original x where False
            return torch.where(mask, input_ids, x)
        # batch['Style'] = batch['Style'].map({
        #     "Gandhi":0,
        #     "Mandela":1
        # })
        # batch['Style'] = 
        style_labels = list(batch['Style'])
        target_labels = list(batch['Target'])
        
        
        sampler = Sampler(cfg, device=args.device)
        texts = sampler.sample(tokenizer, model, graph, noise, style_labels, target_labels, batch_size=batch_size, steps=args.steps, show_intermediate=args.show_intermediate, projector=batch_proj_fun)
        counterspeech_list.extend(texts)
        # for text in texts:
        #     # Use tokenizer to count tokens in each generated output
        #     total_generated_tokens += len(tokenizer.encode(text))
    # print(counterspeech_list)
        df.loc[batch.index, 'CS'] = texts
        
        df.to_csv(args.save_dir, index=False)
    
    end_time = time.time()
    sampling_time = end_time - sampling_start_time

    # Calculate tokens per second
    tokens_per_second = total_generated_tokens / sampling_time if sampling_time > 0 else 0

    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"Sampling time: {sampling_time:.2f} seconds")
    print(f"Total generated tokens: {total_generated_tokens}")
    print(f"Tokens per second (TPS) for {batch_size} rows: {tokens_per_second:.2f}")
    # def proj_fun(x):
        
    #     x[:, input_locs] = input_ids
    #     return x    

if __name__=="__main__":
    main()