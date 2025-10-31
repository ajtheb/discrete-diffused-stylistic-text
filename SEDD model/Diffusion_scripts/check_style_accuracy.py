import os
import sys
sys.path.append('/home/aswini/Forth_project_on_Person_inspired_CS/Diffusion_Language_Modeling/')
from sedd.models.sedd_BERT3 import SEDD_BERT_CLIME
import yaml
from transformers import BertTokenizer
import pandas as pd
import torch
from transformers import (
                          AdamW)
import numpy as np
from sklearn.metrics import f1_score,confusion_matrix
from torch.utils.data import DataLoader, Subset, TensorDataset

def flat_accuracy(preds, labels):
    preds_flat = np.array(preds).flatten()
    labels_flat = np.array(labels).flatten()
    return np.sum(preds_flat == labels_flat) / len(labels_flat)
# Function to calculate the f1 score of our predictions vs labels
def f1_value(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return f1_score(labels_flat,pred_flat, zero_division=0)
def calculate_category_accuracy(bert_model, tokenizer, sentences, labels):
    
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
        roberta_encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 256,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        # Add the encoded sentence to the list.    
        input_ids.append(roberta_encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(roberta_encoded_dict['attention_mask'])
        
        # collecting sentence_ids
        sentence_ids.append(counter)
        counter  = counter + 1
        
    # Convert the lists into tensors.
    roberta_input_ids = torch.cat(input_ids, dim=0)
    roberta_attention_masks = torch.cat(attention_masks, dim=0)
    print(labels)
    labels = torch.tensor(labels)
    sentence_ids = torch.tensor(sentence_ids)

    # print("Tokenization done!!")


    roberta_infer_dataset = TensorDataset(roberta_input_ids, roberta_attention_masks, labels)
    
    batch_size = 32

    roberta_infer_dataloader = DataLoader(
                roberta_infer_dataset,  # The training samples.
                # sampler = RandomSampler(roberta_train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )
    
    
    bert_model.to(device)
    bert_model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_f1_score = 0
    total_eval_loss = 0

    preds_all = []
    labels_all = []
    
    # gandhi_probs = []
    # mandela_probs = []
    gold_probs = []
    
    # Evaluate data for one epoch
    for batch in roberta_infer_dataloader:

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            outputs = bert_model(
                b_input_ids,
                attention_mask=b_input_mask,
                return_dict=False
            )
            logits = outputs[0]
            # probs = torch.softmax(logits, dim=1)
            
            # Collect probabilities for both classes
            # gandhi_probs.extend(probs[:, 0].cpu().numpy())  # Assuming class 0 is Gandhi
            # mandela_probs.extend(probs[:, 1].cpu().numpy())
        # for i in range(len(b_labels)):
        #     gold_label = b_labels[i].item()
        #     gold_prob = probs[i, gold_label].cpu().item()
        #     gold_probs.append(gold_prob)
        # Accumulate the validation loss.
        # total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        # print(logits)
        label_ids = b_labels.to('cpu').numpy()
        
        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        # print(logits)
        # f_acc = flat_accuracy(logits, label_ids)
        # total_eval_accuracy += f_acc
        # total_f1_score += f1_value(logits,label_ids)
        
        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        preds_all.extend(pred_flat)
        labels_all.extend(labels_flat)
        
        # avg_gandhi = np.mean(gandhi_probs)
        # avg_mandela = np.mean(mandela_probs)
        # avg_gold_prob = np.mean(gold_probs)
        
    # print("avg prob:: ", avg_gold_prob)
    # print("avg mandela:: ", avg_mandela)
    print(labels_all)
    print(preds_all)
    score = flat_accuracy(labels_all, preds_all)

    return score

with open("configs/config.yaml", 'r') as f:
    cfg = yaml.full_load(f)
    
device='cuda'
model = SEDD_BERT_CLIME(cfg)
tokenizer = BertTokenizer.from_pretrained('/home/models/bert-base-uncased') 
score_model = SEDD_BERT_CLIME(cfg).to(device)

bert_model = score_model.personality_bert

df = pd.read_csv('/home/aswini/Forth_project_on_Person_inspired_CS/Diffusion_Language_Modeling/Diffusion_scripts/input/Train_new.csv')

sentences = df['Counterspeech']
df['Style'] = df['Style'].map({"Gandhi":0, "Mandela":1}) 
labels = list(df['Style'])

score = calculate_category_accuracy(bert_model, tokenizer, sentences, labels)

print(score)