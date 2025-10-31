import os
import sys
sys.path.append('/home/aswini/Forth_project_on_Person_inspired_CS/Diffusion_Language_Modeling/')
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast
from transformers import BertTokenizer
from sedd.models.noise import LogLinearNoise
from sedd.models.sedd import SEDD
from sedd.models.sedd_BERT2 import SEDD_BERT, VectorQuantizer
from sedd.models.sedd_BERT_ia3 import SEDD_BERT_IA3
from sedd.models.sedd_BERT3 import SEDD_BERT_CLIME, SEDD_BERT_COGENT
from sedd.models.sampler import Sampler
from sedd.models.graph import AbsorbingGraph
from sedd.trainer.trainer import Trainer
from sedd.eval.evaluator import Evaluator
import pandas as pd
import argparse
from aim import Run
from sklearn.preprocessing import LabelEncoder


df_train = pd.read_csv("Diffusion_scripts/input/Train_new.csv")
df_eval = pd.read_csv("Diffusion_scripts/input/Eval_new.csv")

# Initialize LabelEncoder
le = LabelEncoder()

# Fit and transform on training data
df_train['Target'] = le.fit_transform(df_train['Target'])

# Transform evaluation data using the same encoder
df_eval['Target'] = le.transform(df_eval['Target'])

df_train['Style'] = df_train['Style'].map({"Gandhi":0, "Mandela":1})
df_eval['Style'] = df_eval['Style'].map({"Gandhi":0, "Mandela":1})
# df_train = df_train[df_train['Style']=='Gandhi']
# df_train = df_train.reset_index()

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, hatespeech_data, style_data, counterspeech_data, target_data, tokenizer, model_type):
        self.hatespeech_data = hatespeech_data
        self.style_data = style_data
        self.counterspeech_data = counterspeech_data
        self.target_data = target_data
        self.tokenizer = tokenizer
        self.model_type = model_type

    def __len__(self):
        return len(self.hatespeech_data)

    def __getitem__(self, idx):
        hatespeech = self.hatespeech_data[idx]
        style = self.style_data[idx]
        counterspeech = self.counterspeech_data[idx]
        target = self.target_data[idx]

        # input_text = f"[CLS] {hatespeech} [SEP] {style} [SEP] {target} [SEP] {counterspeech} [SEP]"
        # input_text = f"[CLS] {counterspeech} [SEP] "
        # input_text = f"[CLS] {hatespeech} [SEP] {counterspeech} [SEP]"
        
        if(style==0):
            style_text = 'Gandhi'
        else:
            style_text = 'Mandela'
            
        
        target_text = le.inverse_transform([target])
        
        # Dynamic input construction based on model type
        if self.model_type == 'bert_clime':
            input_text = f"[CLS] {counterspeech} [SEP] {counterspeech} [SEP]"
        if self.model_type == 'normal':
            # Original format for other models
            input_text = f"{hatespeech} {style_text} {target_text[0]} {counterspeech}"
            # Original format for other models
        else:
            # Original format for other models
            input_text = f"[CLS] {hatespeech} [SEP] {style_text} [SEP] {target_text[0]} [SEP] {counterspeech} [SEP]"
        
        # print(input_text)
        encoded_input = self.tokenizer(input_text,
                                       padding='max_length',
                                       truncation=True,
                                       max_length=256,
                                       return_tensors='pt')
        return {
            'input_ids': encoded_input['input_ids'].squeeze(0),
            'hatespeech': hatespeech,
            'style': style,
            'counterspeech': counterspeech,
            'target': target
        }
        
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total:,} | Trainable: {trainable:,}"
def main():
    parser = argparse.ArgumentParser(description="Train codebooks with diffusion")
    parser.add_argument("--model_type", type=str, required=True, help="type (normal/bert) ")
    parser.add_argument("--save_dir", type=str, required=True, help="Checkpoint path ")
    parser.add_argument("--vector_quantization", type=int,default=0, required=True, help="Flag for vector quantization(0/1)")
    parser.add_argument("--clime_model", type=str, help="Checkpoint path ")
    
    
    args = parser.parse_args()
    
    model_type = args.model_type
    save_dir_path = args.save_dir
    vq_flag = args.vector_quantization
    
    # Load configuration
    with open("configs/config.yaml", 'r') as f:
        cfg = yaml.full_load(f)
    
    # set use_style_embedding, flag for using codebook lookup
    if(vq_flag):
        cfg['model']['use_style_embedding'] = True
    else:
        cfg['model']['use_style_embedding'] = False
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizers for corresponding model
    if(model_type=='normal'):
        # Initialize tokenizer
        print(f"Model type is = Normal")
        print(f"Using GPT tokenizer.................")
        tokenizer = GPT2TokenizerFast.from_pretrained('/home/models/gpt2')
        tokenizer.pad_token = '<PAD>'
        # for gpt2
        score_model = SEDD(cfg, tokenizer.vocab_size).to(device)
        # cfg['model']['style_dim'] = 50258
    elif(model_type=='bert'):
        # for bert
        print(f"Model type is = bert")
        print(f"Using bert-base-uncased tokenizer.................")
        tokenizer = BertTokenizer.from_pretrained('/home/models/bert-base-uncased') 
        score_model = SEDD_BERT(cfg).to(device)
        # cfg['model']['style_dim'] = 1024
        
    elif(model_type=='bert_ia3'):
        # for bert
        print(f"Model type is = bert_ia3")
        print(f"Using bert-base-uncased tokenizer.................")
        tokenizer = BertTokenizer.from_pretrained('/home/models/bert-base-uncased') 
        score_model = SEDD_BERT_IA3(cfg).to(device)
    
    elif(model_type=='bert_clime'):
        # for bert
        print(f"Model type is = bert_clime")
        print(f"Using bert-base-uncased tokenizer.................")
        tokenizer = BertTokenizer.from_pretrained('/home/models/bert-base-uncased') 
        score_model = SEDD_BERT_CLIME(cfg).to(device)
    
    elif(model_type=='bert_cogent'):
        print(f"Model type is = bert_cogent")
        #print(f"Using bert-base-uncased tokenizer.................")
        clime_model = SEDD_BERT_CLIME(cfg)
        # Load trained codebooks and other modules
        model_file = os.path.join(args.clime_model, "checkpoint.pth")
        loaded_state = torch.load(model_file, map_location='cuda')
        
        clime_model.load_state_dict(loaded_state, strict=False)
        # Freeze all parameters except fusion and bert module
        for param in clime_model.parameters():
            param.requires_grad_(False)
        clime_model.eval()
        # Unfreeze PerFuMe fusion module
        for param in clime_model.fusion.parameters():
            param.requires_grad_(True)
        
        for param in clime_model.bert.parameters():
            param.requires_grad_(True)
        # for bert
        tokenizer = BertTokenizer.from_pretrained('/home/models/bert-base-uncased') 
        score_model = SEDD_BERT_COGENT(cfg, clime_model).to(device)
    
    
    print("\nModel Parameters:", count_parameters(score_model))
    
    cfg['model']['model_type'] = args.model_type
    
    # Prepare your dataset (replace with actual loading logic)
    hatespeech_data_train= df_train['Hatespeech']  # Load your hatespeech data here
    style_data_train = df_train['Style']      # Load your intent data here (e.g., Informative, Questioning)
    counterspeech_data_train = df_train['Counterspeech']  # Load your counterspeech data here
    target_data_train = df_train['Target']
    
    hatespeech_data_eval = df_eval['Hatespeech']  # Load your hatespeech data here
    style_data_eval = df_eval['Style']      # Load your intent data here (e.g., Informative, Questioning)
    counterspeech_data_eval = df_eval['Counterspeech']  # Load your counterspeech data here
    target_data_eval = df_eval['Target']
    
    
    dataset_train = CustomDataset(hatespeech_data_train, style_data_train, counterspeech_data_train,target_data_train,  tokenizer, model_type)
    dataset_eval = CustomDataset(hatespeech_data_eval, style_data_eval, counterspeech_data_eval, target_data_eval, tokenizer,model_type) 
    train_loader = DataLoader(dataset_train, batch_size=cfg['training']['batch_size'], shuffle=True)
    eval_loader = DataLoader(dataset_eval, batch_size=cfg['training']['batch_size'], shuffle=True)

    
    # Initialize model and noise schedule
    graph = AbsorbingGraph(tokenizer.vocab_size)
    
    noise = LogLinearNoise().to(device)
    style_quantizer = score_model.style_quantizer
    
    run = Run()
    run["hparams"] = cfg
    def eval(state):
        evaluator = Evaluator(eval_loader, run, cfg, device=device)
        return evaluator.evaluate(state, cfg['model']['use_style_embedding'])
    
    def sample(state):
        step = state['step']
        model = state['model']
        graph = state['graph']
        noise = state['noise']

        sampler = Sampler(cfg)
        texts = sampler.sample(tokenizer,
                                model,
                                graph,
                                noise,
                                #style_labels,
                                #target_labels,
                                batch_size=cfg['training']['batch_size'],
                                steps=cfg['sampling']['steps']
                                )

        for i in range(3):
            print(f"***********************************************************")
            print(texts[i])
            print(f"***********************************************************")


    # Set up trainer and evaluator
    trainer = Trainer(
        run=None,  # Replace with actual run object if needed for tracking
        model=score_model,
        graph=graph,
        noise=noise,
        config=cfg,
        style_quantizer=style_quantizer, # optional
        eval_callback=eval,  # Implement evaluation callback if needed
        sample_callback=sample,  # Implement sampling callback if needed
        device=device,
        checkpoint_dir=save_dir_path  # Adjust as necessary for saving checkpoints
    )
    # Train the model
    trainer.train(train_loader)

if __name__ == "__main__":
    main()
