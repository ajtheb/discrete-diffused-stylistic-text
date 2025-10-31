import os
import sys
sys.path.append('/home/aswini/Forth_project_on_Person_inspired_CS/Diffusion_Language_Modeling/')
import torch
import argparse
import yaml
from transformers import BertTokenizer, BertModel
from transformers import GPT2TokenizerFast

from sedd.models.sedd_BERT2 import SEDD_BERT
from sedd.models.sedd import SEDD
from sedd.models.sedd_BERT_ia3 import SEDD_BERT_IA3
from sedd.models.sedd_BERT_rag import SEDD_BERT_rag
from sedd.models.sedd_BERT3 import SEDD_BERT_CLIME,SEDD_BERT_COGENT

from sedd.models.graph import AbsorbingGraph
from sedd.models.noise import LogLinearNoise
from sedd.models.sampler import Sampler
import time
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_type", default="normal", type=str)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--show_intermediate", action='store_true')
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--vector_quantization", type=int,default=0, required=True, help="Flag for vector quantization(0/1)")
    parser.add_argument("--clime_model", type=str,default="", help="SEDD_BERT_CLIME")
    args = parser.parse_args()
    
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
    context_embedding = None
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
    elif(model_type=='bert_ia3'):
        # for bert
        tokenizer = BertTokenizer.from_pretrained('/home/models/bert-base-uncased') 
        model = SEDD_BERT_IA3(cfg).to(args.device)
        cfg['model']['style_dim'] = 1024
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
        
    elif(model_type=='bert_rag'):
        # for bert
        tokenizer = BertTokenizer.from_pretrained('/home/models/bert-base-uncased') 
        model = SEDD_BERT_rag(cfg).to(args.device)
        cfg['model']['style_dim'] = 1024
        context_text = "An eye for an eye make the whole world blind."
        # context_text = "The 2023 Nobel Prize in Physics was awarded to Pierre Agostini, Ferenc Krausz, and Anne L’Huillier for their work on attosecond pulses."
        bert_model = BertModel.from_pretrained('/home/models/bert-base-uncased').to(args.device)
        bert_model.eval()
        inputs = tokenizer(context_text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(args.device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            context_embedding = outputs.pooler_output  # [1, hidden_size]
    
    df_train = pd.read_csv("Diffusion_scripts/input/Train_new_2.csv")
    df = pd.read_csv("Diffusion_scripts/input/Test_new_2.csv")
    
    le = LabelEncoder()

    # Fit and transform on training data
    df_train['Target'] = le.fit_transform(df_train['Target'])
    
    print("\nLabel Encoder Mappings:")
    for i, class_name in enumerate(le.classes_):
        print(f"{class_name} → {i}")
        
    # tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    # tokenizer.pad_token = '<PAD>'
    
    print("Vocab size: ", tokenizer.vocab_size)
    print("Last token in vocab: ", tokenizer.decode([tokenizer.vocab_size-1]))
    print("Past vocab size added in the vocabulary EMPTY_STRING: ", tokenizer.batch_decode([[tokenizer.vocab_size]]))

    print("Complete prefix: ", args.prefix)
    print("with suffix: ", args.suffix)
    # hs = "Feminism is just an excuse for women to disrespect mens rights."
    prefix = "[CLS] \ud83d\ude02\ud83d\ude02\ud83d\ude02\ud83d\ude02\ud83d\ude02  Defending sand niggers  \ud83d\ude02\ud83d\ude02\ud83d\ude02\ud83d\ude02\ud83d\ude02  You should visit Iran and see how they treat whites \ud83d\ude09  Seeing islam implode takes priority over \"reee muuuuhhhh JOOOOOOOOOOOOOOOOOOOOZZZZZZZZZZZZZ\" [SEP] Gandhi [SEP] Muslims [SEP] "
    # suffix = "As I have always said, ""Be the change you wish to see in the world."" We must recognize that every individual, regardless of their gender, possesses inherent dignity and worth. The notion that women are incapable of doing men's stuff is a misguided and hurtful stereotype. In reality, women are not oppressed, but rather, they are the backbone of our society, contributing in countless ways to our collective well-being. Let us strive to create a world where every person can live with dignity, respect, and compassion, regardless of their gender. By embracing our shared humanity and the power of non-violence and truth, we can build bridges of kindness and foster a society that values the unique contributions of every individual."
    suffix = ""
    # suffix = "As I have always said, ""Be the change you wish to see in the world."" We must recognize that every individual, regardless of their gender, possesses inherent dignity and worth. The notion that women are incapable of doing men's stuff is a misguided and hurtful stereotype. In reality, women are not oppressed, but rather, they are the backbone of our society, contributing in countless ways to our collective well-being. Let us strive to create a world where every person can live with dignity, respect, and compassion, regardless of their gender. By embracing our shared humanity and the power of non-violence and truth, we can build bridges of kindness and foster a society that values the unique contributions of every individual."
    # prefix = ""
    # pre_ids = tokenizer(pre).input_ids
    prefix_ids = tokenizer(prefix).input_ids
    suffix_ids = tokenizer(suffix).input_ids
    input_ids = prefix_ids + suffix_ids
    input_locs = list(range(len(prefix_ids))) + list(range(256-len(suffix_ids), 256))
    
    style_label = [0]# Gandhi - 0 and Mandela - 1
    target_label = [4]

    # print(input_locs)
    # more generaly commands can be defined with something like below:
    # input_ids = [0, 1, 512, 8080, 50256, 20000]
    # input_locs = [5, 6, 19, 20, 1000, 10001]

    input_ids = torch.tensor(input_ids, device="cuda")[None].repeat(args.batch_size, 1)

    def proj_fun(x):
        x[:, input_locs] = input_ids
        return x
    # Load the model onto GPU
    #device = torch.device('cuda')
    # model = SEDD(cfg, tokenizer.vocab_size).to(args.device)
    # model = SEDD_BERT(cfg).to(args.device)
    model_file = os.path.join(args.model, "checkpoint.pth")
    # print(model_file)
    loaded_state = torch.load(model_file, map_location=args.device)
    # filtered_state = {
    #     k: v for k, v in loaded_state.items() 
    #     if k in model.state_dict() and not k.startswith('style_quantizer.')
    # }
    print(model.bert.config.vocab_size)
    model.load_state_dict(loaded_state, strict=False)
    
    # print(model.style_quantizer.embedding.weight)
    # model.load_state_dict(loaded_state)

    # Load the transition graph
    graph = AbsorbingGraph(tokenizer.vocab_size)

    #print(type(graph))
    
    # Load the noise function
    noise = LogLinearNoise().to(args.device)

    # print(context_embedding)
    start_time = time.time()
    sampler = Sampler(cfg, device=args.device)
    texts = sampler.sample(tokenizer, model, graph, noise, style_label, target_label, steps=args.steps, show_intermediate=args.show_intermediate, projector=proj_fun, context_embedding=context_embedding)
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nInference Time: {elapsed:.2f} seconds")
    for i in texts:
        print("="*80)
        print(i)
        print("="*80)

if __name__=="__main__":
    main()