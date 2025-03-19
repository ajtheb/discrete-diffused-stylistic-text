import argparse
import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.translate.gleu_score import sentence_gleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
from collections import Counter
from textstat import flesch_reading_ease
from detoxify import Detoxify
from tabulate import tabulate
import nltk
import torch
import warnings
import scipy
from transformers import (RobertaForSequenceClassification,
                          RobertaTokenizer)
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from evaluate import load
import re
from rapidfuzz import fuzz, process
import json

warnings.filterwarnings('ignore')

def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except Exception as e:
        print(f"Warning: Failed to download NLTK data: {e}")

def tokenize_text(text):
    """Properly tokenize text"""
    try:
        return nltk.word_tokenize(str(text))
    except Exception as e:
        return str(text).split()

def calculate_repetition_rate(text):
    """Calculate repetition rate of words"""
    words = tokenize_text(text)
    if not words:
        return 0
    word_counts = Counter(words)
    return 1 - (len(word_counts) / len(words))

def calculate_diversity(text):
    """
    Calculate lexical diversity with improved validation and penalties
    - Returns 0 for very short texts
    - Applies penalties for repetitive text
    - Considers meaningful word length
    - Accounts for text coherence
    """
    words = tokenize_text(text)
    
    # Return 0 for very short texts
    if len(words) < 3:
        return 0
        
    unique_words = set(words)
    
    # Return 0 if too few unique words
    if len(unique_words) < 2:
        return 0
    
    try:
        # Base diversity score
        base_diversity = len(unique_words) / len(words)
        
        # Apply penalties
        penalties = 1.0
        
        # Length penalty
        ideal_length = 5  # Minimum length for meaningful diversity
        length_penalty = min(len(words) / ideal_length, 1)
        penalties *= length_penalty
        
        # Repetition penalty
        word_counts = Counter(words)
        most_common_count = word_counts.most_common(1)[0][1]
        if most_common_count > len(words) * 0.5:  # If any word appears more than 50% of the time
            penalties *= 0.5
        
        final_score = base_diversity * penalties
        return max(0, min(1, final_score))  # Ensure score is between 0 and 1
        
    except Exception as e:
        return 0

def calculate_novelty(reference, hypothesis):
    """
    Calculate novelty between reference and hypothesis with improved validation
    - Returns 0 for very short texts
    - Considers meaningful differences
    - Applies quality-based penalties
    - Accounts for semantic relevance
    """
    ref_words = set(tokenize_text(reference))
    hyp_words = set(tokenize_text(hypothesis))
    
    # Return 0 for invalid inputs
    if len(hyp_words) < 3 or len(ref_words) < 3:
        return 0
    
    try:
        # Base novelty score
        unique_words = hyp_words - ref_words
        base_novelty = len(unique_words) / len(hyp_words)
        
        # Apply penalties
        penalties = 1.0
        
        # Length penalty
        ideal_length = 5  # Minimum length for meaningful novelty
        length_penalty = min(len(hyp_words) / ideal_length, 1)
        penalties *= length_penalty
        
        # Quality penalty for hypothesis
        hyp_diversity = len(set(tokenize_text(hypothesis))) / len(tokenize_text(hypothesis))
        if hyp_diversity < 0.5:  # If hypothesis has low diversity
            penalties *= hyp_diversity
        
        # Semantic relevance penalty
        common_words = hyp_words.intersection(ref_words)
        if len(common_words) == 0:  # If no words in common, might be off-topic
            penalties *= 0.5
        
        final_score = base_novelty * penalties
        return max(0, min(1, final_score))  # Ensure score is between 0 and 1
        
    except:
        return 0

def safe_divide(x, y):
    """Safely divide numbers"""
    try:
        return x / y if y != 0 else 0
    except:
        return 0
    
def calculate_flesch_reading_ease(text):
    """
    Calculate Flesch Reading Ease with validation and penalties
    - Returns 0 for very short texts
    - Applies penalties for repetitive or meaningless text
    - Caps the maximum score at 100
    - Applies length-based scaling
    """
    words = text.split()
    
    # Return 0 for very short texts
    if len(words) < 3:
        return 0
        
    # Check for repetitive text
    unique_words = set(words)
    if len(unique_words) < 3:
        return 0
    
    try:
        base_score = flesch_reading_ease(text)
        
        # Cap the score at 100
        base_score = min(base_score, 100)
        
        # Apply penalties based on text characteristics
        penalties = 1.0
        
        # Length penalty
        ideal_length = 10  # Ideal minimum length for meaningful text
        length_penalty = min(len(words) / ideal_length, 1)
        penalties *= length_penalty
        
        # Diversity penalty
        diversity_ratio = len(unique_words) / len(words)
        if diversity_ratio < 0.5:  # If less than 50% words are unique
            penalties *= diversity_ratio
        
        # Calculate final score with penalties
        final_score = base_score * penalties
        
        return max(0, min(100, final_score))  # Ensure score is between 0 and 100
        
    except:
        return 0

def calculate_toxicity(text, toxicity_model):
    """Calculate toxicity with validation"""
    if len(text.split()) < 3:
        return 1.0
    try:
        return toxicity_model.predict(text)['toxicity']
    except:
        return 1.0

def calculate_meteor_score(reference, hypothesis):
    """
    Calculate METEOR score with improved validation and penalties
    - Returns 0 for very short or nonsensical texts
    - Applies penalties for repetitive content
    - Considers meaningful length
    """
    try:
        ref_tokens = tokenize_text(reference)
        hyp_tokens = tokenize_text(hypothesis)
        
        # Return 0 for very short texts
        if len(hyp_tokens) < 3 or len(ref_tokens) < 3:
            return 0
            
        # Check for repetitive content
        unique_hyp_tokens = set(hyp_tokens)
        if len(unique_hyp_tokens) < 2:
            return 0
        
        # Calculate base METEOR score
        base_score = meteor_score([ref_tokens], hyp_tokens)
        
        # Apply penalties
        penalties = 1.0
        
        # Length penalty
        ideal_length = 5
        length_penalty = min(len(hyp_tokens) / ideal_length, 1)
        penalties *= length_penalty
        
        # Diversity penalty
        diversity_ratio = len(unique_hyp_tokens) / len(hyp_tokens)
        if diversity_ratio < 0.5:  # If less than 50% words are unique
            penalties *= diversity_ratio
        
        final_score = base_score * penalties
        return max(0, min(1, final_score))
        
    except:
        return 0

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

def calculate_category_accuracy(sentences, labels):
    
    roberta_tokenizer = RobertaTokenizer.from_pretrained("/home/models/roberta-large")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    roberta_input_ids = []
    roberta_attention_masks = []
    sentence_ids = []
    counter = 0

    ### Tokenization ##
    # For every sentence...
    for sent in sentences:
        roberta_encoded_dict = roberta_tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 150,           # Pad & truncate all sentences.
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

    # print("Tokenization done!!")


    roberta_infer_dataset = TensorDataset(roberta_input_ids, roberta_attention_masks, labels)
    
    batch_size = 32

    roberta_infer_dataloader = DataLoader(
                roberta_infer_dataset,  # The training samples.
                # sampler = RandomSampler(roberta_train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )
    
    roberta_model = RobertaForSequenceClassification.from_pretrained("/home/models/roberta-large", # 12-layer, 768-hidden, 12-heads, 125M parameters RoBERTa using the BERT-base architecture
                                                                    num_labels = 2, # The number of output labels--2 for binary classification.
                                                                                    # You can increase this for multi-class tasks.   
                                                                    output_attentions = False, # Whether the model returns attentions weights.
                                                                    output_hidden_states = False # Whether the model returns all hidden-states.
                                                                )
    roberta_model.to(device)
    checkpoint_path = '/home/aswini/Forth_project_on_Person_inspired_CS/author_style_classifier/saved_models/model_gandhi_style_epoch_2_fold9.pth'
    roberta_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # roberta_model.load_state_dict(checkpoint['model_state_dict'])
    roberta_model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_f1_score = 0
    total_eval_loss = 0

    # Evaluate data for one epoch
    for batch in roberta_infer_dataloader:

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            (loss, logits) = roberta_model(b_input_ids, 
#                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                            return_dict= False)

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        # print(logits)
        label_ids = b_labels.to('cpu').numpy()
        
        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        # print(logits)
        f_acc = flat_accuracy(logits, label_ids)
        total_eval_accuracy += f_acc
        total_f1_score += f1_value(logits,label_ids)


    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(roberta_infer_dataloader)
    # print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    
    avg_f1_score = total_f1_score / len(roberta_infer_dataloader)
    # print("  F1 score: {0:.2f}".format(avg_f1_score))

    return avg_f1_score

def calculate_next_sentence_prediction(sentence1, sentence2, style, target):
    roberta_tokenizer = RobertaTokenizer.from_pretrained("/home/models/roberta-large")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    sentence1 = style + "[SEP]" + target + "[SEP]" + sentence1

    roberta_input_ids = []
    roberta_attention_masks = []
    
    for sent_index in range(len(sentence1)):
        roberta_encoded_dict = roberta_tokenizer.encode_plus(
            sentence1[sent_index],
            sentence2[sent_index],
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        roberta_input_ids.append(roberta_encoded_dict['input_ids'])
        roberta_attention_masks.append(roberta_encoded_dict['attention_mask'])
    
    roberta_input_ids = torch.cat(roberta_input_ids, dim=0)
    roberta_attention_masks = torch.cat(roberta_attention_masks, dim=0)

    roberta_infer_dataset = TensorDataset(roberta_input_ids, roberta_attention_masks)
    batch_size = 32
    roberta_infer_dataloader = DataLoader(roberta_infer_dataset, batch_size=batch_size)

    roberta_model = RobertaForSequenceClassification.from_pretrained("/home/models/roberta-large", num_labels=2)
    roberta_model.load_state_dict(torch.load('/home/aswini/Forth_project_on_Person_inspired_CS/next sentence prediction/trial_models/model_nsp_8_fold4.pth', map_location=device))
    roberta_model.to(device)
    roberta_model.eval()

    total_prob_label1 = 0
    num_samples = 0

    with torch.no_grad():
        for batch in roberta_infer_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            logits = roberta_model(b_input_ids, attention_mask=b_input_mask, return_dict=False)[0]
            probs = F.softmax(logits, dim=1)[:, 1]  # Probability of label 1
            total_prob_label1 += probs.sum().item()
            num_samples += probs.shape[0]
    
    avg_prob_label1 = total_prob_label1 / num_samples
    print(f"Average Probability of Label 1: {avg_prob_label1:.4f}")
    return avg_prob_label1


def clean(text):
    # Fix encoding issues and remove newline characters
    text = text.replace("â€™", "'")
    text = text.replace("\n", "")
    # Convert single quotes around quoted text to double quotes
    text = re.sub(r"(?<!\w)'(.*?)'(?!\w)", r'"\1"', text)

    if text and text[0] == '"':
        text = text[1:]
    if text and text[-1] == '"':
        text = text[:-1]

    return text

"""
Return quote related metrics
metrics: 
1.Quote Fuzz score: Average fuzz score calculated using edit distance(fuzz score) between extracted quote and most similar original quote.
    Score=(1-(Levenshtein Distance)/(max(len(s1),len(s2)))*100
    Range = [0-100]
2. Quote presence: how many counterspeeches contain atleast 1 valid extracted quote?
    Range = [0-1]
"""
def fuzzy_match(mandela_quotes, gandhi_quotes, counterspeeches, style, threshold=80):
    data = []
    total_score = 0
    total_valid_cs = 0
    extracted_quotes_list = []

    for index, cs in enumerate(counterspeeches):
        # Alternate between the two authors based on index:
        if(style[index]=='Mandela'):
            quotes = mandela_quotes
        else:
            quotes = gandhi_quotes
        # Extract the quotes
        extracted_quotes = re.findall(r'"(.*?)"', cs)
        # Total cs score
        score_cs = 0
        # 
        count_q = 0
        
        q1 = "Quote"
        for quote_i in extracted_quotes:
            if len(quote_i.split()) > 2:
                q1 = quote_i
                # calculate fuzz score with all quotes
                fz_scores = [fuzz.partial_ratio(q, quote_i) for q in quotes]
                max_fuzz_score = max(fz_scores)
                # increase fuzz score
                score_cs += max_fuzz_score
                count_q += 1

        extracted_quotes_list.append(q1)
        if count_q > 0:
            score_cs /= count_q
            total_score += score_cs
            total_valid_cs += 1
        else:
            score_cs = None
        
        data.append([cs, extracted_quotes, score_cs])

    df = pd.DataFrame(data, columns=['Counterspeech', 'Quote', 'Max_Fuzz_Score'])
    df.to_csv('cs_fuzzscore.csv', index=False)

    try:
        quote_avg_fuzz = total_score / len(counterspeeches)
        quote_presence = total_valid_cs / len(counterspeeches)
    except Exception as e:
        print("Exception during metric calculation:", e)
        quote_avg_fuzz = quote_presence = None

    print("Quote Average Fuzz:", quote_avg_fuzz)
    print("Quote Presence Accuracy:", quote_presence)

    return (quote_avg_fuzz, quote_presence)

def compute_metrics(data, batch_size=32, finetune=False):
    """Compute all metrics for the dataset"""
    metrics = []
    smoother = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Initialize models
    try:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        toxicity_model = Detoxify('unbiased')
    except Exception as e:
        print(f"Warning: Failed to load models: {e}")
        return []
    # print(data.head())
    # data = data[:10]
    # BLUE_score = load("bleu")

    # bleu_score = BLUE_score.compute(predictions=data['CS'], references=data['Reference CS'])
    # print("bleu",bleu_score)

    # meteor = load("meteor")

    # meteor_score = meteor.compute(predictions=data['CS'], references=data['Reference CS'])
    # print("meteor score", meteor_score )

    for idx, row in data.iterrows():
        try:
            reference = str(row['Reference CS'])
            hypothesis = str(row['CS'])
            # hatespeech = str(row['Hatespeech'])

            # Stricter input validation
            if len(hypothesis.split()) < 3 or len(set(hypothesis.split())) < 3:
                print(f"Warning: Invalid output at row {idx}: {hypothesis}")
                metrics.append([0] * 13)
                continue

            if len(set(hypothesis.split())) < 2:
                print(f"Warning: Output lacks diversity at row {idx}: {hypothesis}")
                metrics.append([0] * 13)
                continue

            # Tokenize texts
            ref_tokens = tokenize_text(reference)
            hyp_tokens = tokenize_text(hypothesis)

            # Calculate BLEU
            try:
                bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoother)
            except:
                bleu = 0

            # ROUGE with stricter thresholds
            try:
                rouge_scores = rouge.score(reference, hypothesis)
                # Apply penalty for short or repetitive outputs
                length_penalty = min(len(hypothesis.split()) / len(reference.split()), 1)
                rouge1 = rouge_scores['rouge1'].fmeasure * length_penalty
                rouge2 = rouge_scores['rouge2'].fmeasure * length_penalty
                rougeL = rouge_scores['rougeL'].fmeasure * length_penalty
            except:
                rouge1 = rouge2 = rougeL = 0

            # Calculate METEOR
            try:
                meteor = calculate_meteor_score(reference, hypothesis)
            except:
                meteor = 0

            # Calculate GLEU
            try:
                gleu = sentence_gleu([ref_tokens], hyp_tokens)
            except:
                gleu = 0

            # Calculate other metrics
            rep_rate = calculate_repetition_rate(hypothesis)
           
            flesch = calculate_flesch_reading_ease(hypothesis)
            
            # Calculate semantic similarity
            try:
                embeddings = model.encode([reference, hypothesis])
                cosim = 1 - scipy.spatial.distance.cosine(embeddings[0], embeddings[1])

            except:
                cosim = 0

            # Modified BERT Score calculation
            try:
                with torch.no_grad():
                    P, R, F1 = bert_score([hypothesis], [reference], 
                                        lang='en',
                                        model_type='microsoft/deberta-xlarge-mnli')
                    # Apply penalty for short outputs
                    bert_f1 = F1.mean().item() * length_penalty
            except:
                bert_f1 = 0


            # Calculate toxicity
            toxicity = calculate_toxicity(hypothesis, toxicity_model)

            novelty = calculate_novelty(reference, hypothesis)
            diversity = calculate_diversity(hypothesis)

            # Validate metrics
            metrics_list = [
                max(0, min(1, bleu)),
                max(0, min(1, rouge1)),
                max(0, min(1, rouge2)),
                max(0, min(1, rougeL)),
                max(0, min(1, meteor)),
                max(0, min(1, gleu)),
                max(0, min(1, rep_rate)),
                max(0, min(100, flesch)),
                max(0, min(1, cosim)),
                max(0, min(1, bert_f1)),
                max(0, min(1, toxicity)),
                max(0, min(1, novelty)),
                max(0, min(1, diversity))
            ]

            metrics.append([round(x, 4) for x in metrics_list])

        except Exception as e:
            print(f"Warning: Error processing row {idx}: {e}")
            metrics.append([0] * 13)
    

    # counterspeech = data['CS']
    # hatespeech = data['Hatespeech']
    # style = data['Style']
    # target = data['Target']

    # cleaned_cs = []
    # import re
    # for cs in counterspeech:
    #     cleaned_text = re.sub(r'"[^"]*"', '', cs)
    #     cleaned_cs.append(cleaned_text)
    # # print(cleaned_cs)
    # # Calculate next sentence predictionn score( style and target consistent) 
    # nsp_score = calculate_next_sentence_prediction(hatespeech, cleaned_cs, style, target)

    # # Calculate next sentence predictionn score( style and target consistent)
    # if(not finetune):
    #     labels = [0,1]*(len(counterspeech)//2) 
    # else:
    #     labels = list(data['Style'].map({"Gandhi":0, "Mandela":1}))
    
    # # cleaned_cs = ['non-violence']
    # # labels = [0]
    # style_score = calculate_category_accuracy(cleaned_cs, labels)
    # print("style score : ", style_score)
    # print("nsp score : ", nsp_score)
    
    
    return metrics

def clean(text):
    # Fix encoding issues and remove newline characters
    text = text.replace("â€™", "'")
    text = text.replace("\n", "")
    # Convert single quotes around quoted text to double quotes
    text = re.sub(r"(?<!\w)'(.*?)'(?!\w)", r'"\1"', text)

    if text and text[0] == '"':
        text = text[1:]
    if text and text[-1] == '"':
        text = text[:-1]

    return text

def compute_metrics2(data, finetune=False):
    counterspeech = data['CS']
    hatespeech = data['Hatespeech']
    style = data['Style']
    target = data['Target']
    
    counterspeech = counterspeech.apply(clean)
    style_list = list(style)
    cs_list = list(counterspeech)
    
    style_score = nsp_score =  quote_avg_fuzz = quote_presence = None
    
    cleaned_cs = []
    import re
    for cs in counterspeech:
        cleaned_text = re.sub(r'"[^"]*"', '', cs)
        cleaned_cs.append(cleaned_text)
    # print(cleaned_cs)
    # Calculate next sentence predictionn score( style and target consistent) 
    nsp_score = calculate_next_sentence_prediction(hatespeech, cleaned_cs, style, target)

    # Calculate next sentence predictionn score( style and target consistent)
    if(not finetune):
        labels = [0,1]*(len(counterspeech)//2) 
    else:
        labels = list(data['Style'].map({"Gandhi":0, "Mandela":1}))
    
    style_score = calculate_category_accuracy(cleaned_cs, labels)
    # style_score = None
    # sc = calculate_category_accuracy(['self introspection, self purification, self-examination, self-reflection, soul-searching, self-contemplation, and self-scrutiny'], [0])
    # Non violence, Peace, resistance, quiet, calming, unaggressive
    print("style score : ", style_score)
    print("nsp score : ", nsp_score)
    
    # Load quotes data.
    with open("merged_quotes.json", "r", encoding="utf-8") as f:
        quotes_data = json.load(f)

    mandela_quotes = quotes_data['authors']['Nelson Mandela']['quotes']
    gandhi_quotes = quotes_data['authors']['Mahatma Gandhi']['quotes']
    print("Number of Gandhi quotes", len(set(gandhi_quotes)))
    print("Number of Mandela quotes", len(set(mandela_quotes)))

    # start_time = time.time()
    quote_avg_fuzz, quote_presence = fuzzy_match(mandela_quotes, gandhi_quotes, cs_list, style_list)
    # end_time = time.time()

    return (style_score, nsp_score, quote_avg_fuzz, quote_presence)
    
    
def main():
    parser = argparse.ArgumentParser(description="Compute metrics for text generation evaluation")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--fine_tune", type=int, default=0, help="fine tuned model output?")

    args = parser.parse_args()
    print(args.input_file)
    # Download required NLTK data
    download_nltk_data()
    # Read data
    try:
        data = pd.read_csv(args.input_file, encoding="ISO-8859-1")
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    print(data.shape)
    try:
        gt_df = pd.read_csv('input/Person_specific_counterspeech3.csv')
        test_size = 0.2
        random_seed = 42
        print(args.fine_tune)
        if(args.fine_tune):
            train_df, eval_df = train_test_split(gt_df, test_size=test_size, random_state=random_seed)
            print(eval_df.shape)
            data['Reference CS'] = list(eval_df['Counterspeech'])
            data['Hatespeech'] = list(eval_df['Hatespeech'])
            data['Style'] = list(eval_df['Style'])
            data['Target'] = list(eval_df['Target'])
        else:
            data['Reference CS'] = gt_df['Counterspeech']
            data['Hatespeech'] = gt_df['Hatespeech']
            data['Style'] = gt_df['Style']
            data['Target'] = gt_df['Target']
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    # # Check
    # data['CS'] = data['Reference CS']
    print(data.head())
    print(data.isnull().sum())
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(data, args.batch_size, finetune= args.fine_tune)
    # metrics = None
    additional_metrics = compute_metrics2(data, finetune= args.fine_tune)
    
    
    if not metrics:
        print("Error: metrics")
        return

    # Define headers
    headers = [
        "BLEU", "ROUGE1", "ROUGE2", "ROUGEL", "METEOR", "GLEU",
        "Repetition Rate", "Flesch Reading Ease", "CoSIM",
        "BERT Score", "Toxicity", "Novelty", "Diversity"
    ]

    # Display results
    print("\nEvaluation Metrics:")
    print(tabulate(metrics, headers=headers, tablefmt="grid"))

    # Calculate and display averages
    averages = np.mean(metrics, axis=0)
    print("\nAverage Metrics:")
    print(tabulate([averages], headers=headers, tablefmt="grid"))
    
    print("style score:", additional_metrics[0])
    print("NSP score:", additional_metrics[1])
    print("Quote Avg Fuzz:", additional_metrics[2])
    print("Quote Presence:", additional_metrics[3])
    
    # Save results to CSV
    try:
        results_df = pd.DataFrame(metrics, columns=headers)
        output_file = args.input_file.rsplit('.', 1)[0] + '_metrics.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    main()