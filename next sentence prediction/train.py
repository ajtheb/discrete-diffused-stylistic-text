import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import KFold
from transformers import (RobertaForSequenceClassification,
                          RobertaTokenizer,
                          AdamW)
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
import time
import datetime

import numpy as np
from sklearn.metrics import f1_score,confusion_matrix
from collections import Counter
from sklearn.model_selection import StratifiedKFold

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Function to calculate the f1 score of our predictions vs labels
def f1_value(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    # print(Counter(pred_flat))
    # print(Counter(labels_flat))
    return f1_score(labels_flat,pred_flat, average='weighted')

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

df=pd.read_csv("input/nsp_pairs2.csv")

# Text
sentence1 = df['Style'] + "[SEP]" + df['Target'] + "[SEP]" + df['HS']
sentence2 = df['CS']
style = df['Style']
# Labels
labels = df['Label']

roberta_tokenizer = RobertaTokenizer.from_pretrained("/home/models/roberta-large")
    

# Tokenize all of the sentences and map the tokens to thier word IDs.
roberta_input_ids = []
roberta_attention_masks = []
sentence_ids = []
counter = 0

# For every sentence, pairs
for sent_index in range(len(sentence1)):
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens. 
    
    roberta_encoded_dict = roberta_tokenizer.encode_plus(
                        sentence1[sent_index],                      # Sentence to encode.
                        sentence2[sent_index],
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 128,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    roberta_input_ids.append(roberta_encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    roberta_attention_masks.append(roberta_encoded_dict['attention_mask'])
    
    # collecting sentence_ids
    sentence_ids.append(counter)
    counter  = counter + 1
    
# Convert the lists into tensors.
roberta_input_ids = torch.cat(roberta_input_ids, dim=0)
roberta_attention_masks = torch.cat(roberta_attention_masks, dim=0)

labels = torch.tensor(labels)
sentence_ids = torch.tensor(sentence_ids)

print("Tokenization done!!")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=777)

label_counts = Counter(labels.numpy())
print("Label Distribution in Dataset:", label_counts)

f1_scores = []

for fold,(idxT,idxV) in enumerate(skf.split(roberta_input_ids.numpy(), labels.numpy())):

    print('#'*25)
    print('### FOLD %i'%(fold+1))
    print('#'*25)

    train_labels = labels.numpy()[idxT]
    val_labels = labels.numpy()[idxV]
    print("y_true distribution:", Counter(train_labels))
    print("y_pred distribution:", Counter(val_labels))
    
    
    # print(idxT,idxV)
    torch.manual_seed(0)
    
    # train
    train_input_ids = roberta_input_ids[idxT]
    train_attention_mask = roberta_attention_masks[idxT]
    train_labels = labels[idxT]
    
    # test
    test_input_ids = roberta_input_ids[idxV]
    test_attention_mask = roberta_attention_masks[idxV]
    test_labels = labels[idxV]
    
    # print(train_input_ids, train_attention_mask, train_labels)
    roberta_train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
    roberta_val_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)
    
    batch_size = 32

    roberta_train_dataloader = DataLoader(
                roberta_train_dataset,  # The training samples.
                # sampler = RandomSampler(roberta_train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

    roberta_validation_dataloader = DataLoader(
                roberta_val_dataset, # The validation samples.
                # sampler = SequentialSampler(roberta_val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )
    
    roberta_model = RobertaForSequenceClassification.from_pretrained("/home/models/roberta-large", # 12-layer, 768-hidden, 12-heads, 125M parameters RoBERTa using the BERT-base architecture
                                                                    num_labels = 2, # The number of output labels--2 for binary classification.
                                                                                    # You can increase this for multi-class tasks.   
                                                                    output_attentions = False, # Whether the model returns attentions weights.
                                                                    output_hidden_states = False, # Whether the model returns all hidden-states.
                                                                    hidden_dropout_prob = 0.2,
                                                                    attention_probs_dropout_prob = 0.2
                                                                )
        
    roberta_optimizer = AdamW(roberta_model.parameters(),
                  lr = 2e-6, # args.learning_rate - default is 5e-5
                #   eps = 1e-8 # args.adam_epsilon  - default is 1e-8.,
                  weight_decay=0.01
                )
    epochs = 10

    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(roberta_train_dataloader) * epochs

    # Create the learning rate scheduler.
    roberta_scheduler = get_linear_schedule_with_warmup(roberta_optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    
    roberta_training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()
    
   
    roberta_tokenizer = RobertaTokenizer.from_pretrained("/home/models/roberta-large")
    # Tell pytorch to run this model on the GPU.
    roberta_model.cuda()

    min_prev_loss = np.inf
    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the roberta_model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-roberta_model-train-do-in-pytorch)
        roberta_model.train()

        # For each batch of training data...
        for step, batch in enumerate(roberta_train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(roberta_train_dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            roberta_model.zero_grad()        

            # Perform a forward pass (evaluate the roberta_model on this training batch).
            # The documentation for this `roberta_model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/roberta_model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # are given and what flags are set. For our usage here, it returns
            # the loss (because we provided labels) and the "logits"--the roberta_model
            # outputs prior to activation.
            # loss, logits = roberta_model(b_input_ids,attention_mask=b_input_mask,labels=b_labels, return_dict = False)
            
            # Define class weights (modify according to label distribution)
            class_weights = torch.tensor([0.7, 0.3]).to(device)  # Example weights for binary classification

            # Define loss function
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)

            # Perform a forward pass
            outputs = roberta_model(b_input_ids, attention_mask=b_input_mask, return_dict=False)
            logits = outputs[0]  # Extract logits

            # Compute loss using custom loss function
            loss = loss_fn(logits, b_labels)
    #         print(loss, logits, "l")

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(roberta_model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The roberta_optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            roberta_optimizer.step()

            # Update the learning rate.
            roberta_scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(roberta_train_dataloader)            

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the roberta_model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        roberta_model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_f1_score = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in roberta_validation_dataloader:

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using 
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # Get the "logits" output by the roberta_model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                (loss, logits) = roberta_model(b_input_ids, 
    #                                    token_type_ids=None, 
                                       attention_mask=b_input_mask,
                                       labels=b_labels,
                                              return_dict= False)

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            f_acc = flat_accuracy(logits, label_ids)
            total_eval_accuracy += f_acc
            total_f1_score += f1_value(logits,label_ids)


        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(roberta_validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        
        avg_f1_score = total_f1_score / len(roberta_validation_dataloader)
        print("  F1 score: {0:.2f}".format(avg_f1_score))
        

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(roberta_validation_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        roberta_training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

        f1_scores.append(avg_f1_score)

        if(fold==4 and avg_val_loss<min_prev_loss):
            min_prev_loss = avg_val_loss
            #model save
            torch.save(roberta_model.state_dict(), f'trial_models/model_nsp_{epoch_i + 1}_fold{fold}.pth')
    
    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    

# Add this after the training loop
avg_f1_score_all_epochs = sum(f1_scores) / len(f1_scores)
print(f"Average F1 score across all epochs: {avg_f1_score_all_epochs:.4f}")