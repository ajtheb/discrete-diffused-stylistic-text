import os
import yaml
import torch
from torch.utils.data import DataLoader
#from transformers import GPT2TokenizerFast
from transformers import BertTokenizer
from sedd.models.noise import LogLinearNoise
# from sedd.models.sedd import SEDD
from sedd.models.sedd_BERT import SEDD_BERT
from sedd.models.sampler import Sampler
from sedd.models.graph import AbsorbingGraph
from sedd.trainer.trainer import Trainer
from sedd.eval.evaluator import Evaluator
import pandas as pd
#from aim import Run

df_train = pd.read_csv("Diffusion_Language_Modeling/Diffusion_scripts_Gandhi/sample_train_data.csv")

# Initialize tokenizer
# tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
# tokenizer.pad_token = '<PAD>'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, hatespeech_data, intent_data, counterspeech_data, tokenizer):
#         self.hatespeech_data = hatespeech_data
#         self.intent_data = intent_data
#         self.counterspeech_data = counterspeech_data
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.hatespeech_data)

#     def __getitem__(self, idx):
#         hatespeech = self.hatespeech_data[idx]
#         intent = self.intent_data[idx]
#         counterspeech = self.counterspeech_data[idx]

#          # Tokenize the inputs and targets
#         input_text = f"[HATESPEECH] : {hatespeech}, [INTENT] : {intent}, [COUNTERSPEECH] : {counterspeech}"
#         input_ids = self.tokenizer(input_text,
#                                    return_tensors='pt',
#                                    max_length = 256,
#                                    padding='max_length',
#                                    truncation=True
#                                    )
#         return input_ids['input_ids'].squeeze(0)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, hatespeech_data, intent_data, counterspeech_data, tokenizer):
        self.hatespeech_data = hatespeech_data
        self.intent_data = intent_data
        self.counterspeech_data = counterspeech_data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.hatespeech_data)

    def __getitem__(self, idx):
        hatespeech = self.hatespeech_data[idx]
        intent = self.intent_data[idx]
        counterspeech = self.counterspeech_data[idx]

        input_text = f"[CLS] {hatespeech} [SEP] {intent} [SEP] {counterspeech} [SEP]"
        encoded_input = self.tokenizer(input_text,
                                       padding='max_length',
                                       truncation=True,
                                       max_length=256,
                                       return_tensors='pt')
        return encoded_input['input_ids'].squeeze(0)

def main():
    # Load configuration
    with open("Diffusion_Language_Modeling/configs/config.yaml", 'r') as f:
        cfg = yaml.full_load(f)


    
    # Prepare your dataset (replace with actual loading logic)
    hatespeech_data = df_train['Hate Speech']  # Load your hatespeech data here
    intent_data = df_train['Intent']      # Load your intent data here (e.g., Informative, Questioning)
    counterspeech_data = df_train['CounterSpeech']  # Load your counterspeech data here

    dataset = CustomDataset(hatespeech_data, intent_data, counterspeech_data, tokenizer)
    train_loader = DataLoader(dataset, batch_size=cfg['training']['batch_size'], shuffle=True)

    # Initialize model and noise schedule
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = AbsorbingGraph(tokenizer.vocab_size)
    # score_model = SEDD_BERT(cfg, tokenizer.vocab_size).to(device)
    score_model = SEDD_BERT(cfg).to(device)
    noise = LogLinearNoise().to(device)

    # run = Run()
    # run["hparams"] = cfg

    # Set up trainer and evaluator
    trainer = Trainer(
        run=None,  # Replace with actual run object if needed for tracking
        model=score_model,
        graph=graph,
        noise=noise,
        config=cfg,
        eval_callback=None,  # Implement evaluation callback if needed
        sample_callback=None,  # Implement sampling callback if needed
        device=device,
        checkpoint_dir='Diffusion_Outputs/Demo_Semisupervised/checkpoints'  # Adjust as necessary for saving checkpoints
    )

    # Train the model
    trainer.train(train_loader)

if __name__ == "__main__":
    main()
