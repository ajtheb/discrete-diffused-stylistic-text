import os
import sys
sys.path.append('/home/aswini/Forth_project_on_Person_inspired_CS/Diffusion_Language_Modeling/')
from sedd.models.sedd_BERT3 import SEDD_BERT_CLIME, SEDD_BERT_COGENT
import yaml
from transformers import BertTokenizer
import pandas as pd
import torch
from transformers import (
                          AdamW)
import numpy as np
from sklearn.metrics import f1_score,confusion_matrix
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.preprocessing import LabelEncoder

def flat_accuracy(preds, labels):
    preds_flat = np.array(preds).flatten()
    labels_flat = np.array(labels).flatten()
    return np.sum(preds_flat == labels_flat) / len(labels_flat)
# Function to calculate the f1 score of our predictions vs labels
def f1_value(preds, labels):

    return f1_score(labels,preds, zero_division=0, average = 'weighted')

def calculate_category_accuracy(bert_model,classifier_head, tokenizer, sentences, labels):
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []
    sentence_ids = []
    counter = 0

    # sentences = [ remove_content_words(sent) for sent in sentences ]
    ### Tokenization ##
    # For every sentence...
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 256,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
        
        # collecting sentence_ids
        sentence_ids.append(counter)
        counter  = counter + 1
        
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    # print(labels)
    labels = torch.tensor(labels)
    sentence_ids = torch.tensor(sentence_ids)

    # print("Tokenization done!!")


    infer_dataset = TensorDataset(input_ids, attention_masks, labels)
    
    batch_size = 32

    infer_dataloader = DataLoader(
                infer_dataset,  # The training samples.
                # sampler = RandomSampler(roberta_train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )
    
    
    bert_model.to(device)
    bert_model.eval()

    preds_all = []
    labels_all = []
    
    
    # Evaluate data for one epoch
    for batch in infer_dataloader:

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            outputs = bert_model(
                b_input_ids,
                attention_mask=b_input_mask,
                return_dict=True
            )
            hate_semantics = outputs.last_hidden_state
            logits = classifier_head(hate_semantics.mean(dim=1))
            # logits = outputs[0]
            # probs = torch.softmax(logits, dim=1)

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        # print(logits)
        label_ids = b_labels.to('cpu').numpy()
        
        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        preds_all.extend(pred_flat)
        labels_all.extend(labels_flat)

        
    # print("avg prob:: ", avg_gold_prob)
    # print("avg mandela:: ", avg_mandela)
    print(labels_all)
    print(preds_all)
    score = flat_accuracy(labels_all, preds_all)
    f1 = f1_value(labels_all, preds_all)

    return score,f1

import argparse

parser = argparse.ArgumentParser(description='Evaluate model accuracy')
parser.add_argument('--clime_model', required=True, help='Path to CLIME model directory')
parser.add_argument('--model', required=True, help='Path to COGENT model directory')
# parser.add_argument('--data', required=True, help='Path to training data CSV')
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', 
                    help='Device to use for computation (cuda/cpu)')
parser.add_argument("--vector_quantization", type=int,default=0, required=True, help="Flag for vector quantization(0/1)")
args = parser.parse_args()

with open("configs/config.yaml", 'r') as f:
    cfg = yaml.full_load(f)
    
vq_flag = args.vector_quantization
if(vq_flag):
    cfg['model']['use_style_embedding'] = True
else:
    cfg['model']['use_style_embedding'] = False
device='cuda'
clime_model = SEDD_BERT_CLIME(cfg)

model_file = os.path.join(args.clime_model, "checkpoint.pth")
loaded_state = torch.load(model_file, map_location='cuda')
# filtered_state = {
#     k: v for k, v in loaded_state.items() 
#     if k in model.state_dict() and not k.startswith('style_quantizer.')
# }
clime_model.load_state_dict(loaded_state, strict=False)
for param in clime_model.parameters():
    param.requires_grad_(False)
clime_model.eval()



tokenizer = BertTokenizer.from_pretrained('/home/models/bert-base-uncased') 
score_model = SEDD_BERT_COGENT(cfg, clime_model).to(device)

model_file = os.path.join(args.model, "checkpoint.pth")
loaded_state = torch.load(model_file, map_location=args.device)
score_model.load_state_dict(loaded_state, strict=False)

bert_model = score_model.bert

print(score_model.personality_bert)

df_train = pd.read_csv("Diffusion_scripts/input/Train_new_2.csv")
df_test = pd.read_csv("Diffusion_scripts/input/Test_new_2.csv")

# Initialize LabelEncoder
le = LabelEncoder()

# Fit and transform on training data
df_train['Target'] = le.fit_transform(df_train['Target'])

# Transform evaluation data using the same encoder
df_test['Target'] = le.transform(df_test['Target'])


# df = pd.read_csv('/home/aswini/Forth_project_on_Person_inspired_CS/Diffusion_Language_Modeling/Diffusion_scripts/input/Test_new_2.csv')
df = df_test
# df = df[:10]
# f"[CLS] {hatespeech} [SEP] {style_text} [SEP] {target_text[0]} [SEP] {counterspeech} [SEP]"
# sentences = "[CLS] " +  df['Hatespeech'] + " [SEP] " + df['Style'].str + " [SEP] " + df['Target'].str + " [SEP] " + df['Counterspeech'] 
sentences = (
    "[CLS] " + df['Hatespeech'].astype(str) + 
    " [SEP] " + df['Style'].astype(str) + 
    " [SEP] " + df['Target'].astype(str) + 
    " [SEP] " + df['Counterspeech'].astype(str)
)
# df['Style'] = df['Style'].map({"Gandhi":0, "Mandela":1}) 
labels = list(df['Target'])

classifier_head = score_model.target_classifier

score, f1 = calculate_category_accuracy(bert_model, classifier_head, tokenizer, sentences, labels)

print(score, f1)