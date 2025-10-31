"""
Metric file to evaluate the model output.
"""

import argparse
import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.translate.gleu_score import sentence_gleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
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
from sklearn.metrics import f1_score, log_loss
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from evaluate import load
import re
from rapidfuzz import fuzz, process
import json
import logging
import os
os.environ['HF_HOME'] = './huggingface_cache'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_CACHE'] = './huggingface_cache'  # Additional cache variable for redundancy
os.environ['HF_HUB_CACHE'] = './huggingface_cache'  # For hub-related operations
warnings.filterwarnings('ignore')

# Set NLTK data path
def set_nltk_data_path():
    local_nltk_data_path = '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/nltk_data'
    if os.path.exists(local_nltk_data_path):
        nltk.data.path.append(local_nltk_data_path)
    else:
        raise FileNotFoundError(f"NLTK data not found at {local_nltk_data_path}")

def download_nltk_data():
    try:
        set_nltk_data_path()
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
    except LookupError as e:
        raise FileNotFoundError(f"Required NLTK data not found: {e}")

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
    

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    preds_flat = np.array(preds).flatten()
    labels_flat = np.array(labels).flatten()
    return np.sum(preds_flat == labels_flat) / len(labels_flat)

# Function to calculate the f1 score of our predictions vs labels
def f1_value(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    return f1_score(labels_flat,pred_flat, average='weighted')

def remove_content_words(text):
    # Tokenize and tag parts-of-speech
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

    # Filter out nouns(content words)
    content_tags = {'NN', 'NNS'}
    filtered = [word for word, tag in tagged if tag not in content_tags]

    return ' '.join(filtered)

def calculate_category_accuracy(sentences, labels):
    roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Tokenization
    roberta_input_ids, roberta_attention_masks = [], []
    for sent in sentences:
        roberta_encoded_dict = roberta_tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        roberta_input_ids.append(roberta_encoded_dict['input_ids'])
        roberta_attention_masks.append(roberta_encoded_dict['attention_mask'])

    roberta_input_ids = torch.cat(roberta_input_ids, dim=0)
    roberta_attention_masks = torch.cat(roberta_attention_masks, dim=0)

    labels = torch.tensor(labels)

    dataset = TensorDataset(roberta_input_ids, roberta_attention_masks, labels)
    dataloader = DataLoader(dataset, batch_size=32)

    # Load model
    roberta_model = RobertaForSequenceClassification.from_pretrained(
        "roberta-large",
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    roberta_model.to(device)
    checkpoint_path = '/home/aswini/Forth_project_on_Person_inspired_CS/author_style_classifier/saved_models/model_style_epoch_3_fold0_wo_q.pth'
    roberta_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    roberta_model.eval()

    preds_all, labels_all, probs_all, true_label_probs = [], [], [], []

    # Inference loop
    for batch in dataloader:
        b_input_ids, b_input_mask, b_labels = [x.to(device) for x in batch]

        with torch.no_grad():
            (loss, logits) = roberta_model(
                b_input_ids,
                attention_mask=b_input_mask,
                labels=b_labels,
                return_dict=False
            )
            probs = torch.softmax(logits, dim=1)

        preds_all.extend(torch.argmax(logits, dim=1).cpu().numpy())
        labels_all.extend(b_labels.cpu().numpy())
        probs_all.extend(probs.cpu().numpy())

        # store probability for gold label
        for i in range(len(b_labels)):
            true_prob = probs[i, b_labels[i]].item()
            true_label_probs.append(true_prob)

    # Convert to numpy arrays
    preds_all = np.array(preds_all)
    labels_all = np.array(labels_all)
    probs_all = np.array(probs_all)
    true_label_probs = np.array(true_label_probs)

    # Metrics
    accuracy = flat_accuracy(preds_all, labels_all)
    logloss = log_loss(labels_all, probs_all)
    f1 = f1_value(probs_all, labels_all)  # uses your helper
    avg_true_prob = np.mean(true_label_probs)
    print("Mean SC:", avg_true_prob)
    # Save CSV for per-instance true label probability
    df = pd.DataFrame({
        'sentence': sentences,
        'label': labels_all,
        'true_label_prob': true_label_probs
    })
    df.to_csv("SC.csv", index=False)
    
    return accuracy, logloss
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
"""
def fuzzy_match( counterspeeches, style, ref_counterspeech, threshold=80):
    """
    Compute fuzzy matching metrics for counterspeeches against reference quotes and reference counterspeeches.
    Metrics include:
    - Reference-CS Quote Fuzz: Average fuzz score between quotes extracted from generated counterspeeches
      and quotes extracted from reference counterspeeches.

    Args:
        counterspeeches (list): List of generated counterspeeches.
        style (list): List of style labels ('Mandela' or 'Gandhi') for each counterspeech.
        ref_counterspeech (list): List of reference counterspeeches corresponding to generated counterspeeches.
        threshold (int): Minimum fuzz score threshold (default: 80).

    Returns:
        tuple: (quote_avg_fuzz, quote_presence, ref_cs_quote_fuzz)
    """
    data = []
    total_score = 0
    total_valid_cs = 0
    total_ref_cs_score = 0
    extracted_quotes_list = []

    # Validate inputs
    assert len(counterspeeches) == len(style) == len(ref_counterspeech), \
        "Mismatch in lengths of counterspeeches, style, and ref_counterspeech"

    for index, (cs, ref_cs) in enumerate(zip(counterspeeches, ref_counterspeech)):
        
        # Extract quotes from generated counterspeech
        extracted_quotes = re.findall(r'"(.*?)"', cs)
        # Extract quotes from reference counterspeech
        ref_extracted_quotes = re.findall(r'"(.*?)"', ref_cs)
        
        ref_cs_score = 0
        count_q = 0
        q1 = "Quote"  # Default quote if none extracted

        for quote_i in extracted_quotes:
            if len(quote_i.split()) > 2:  # Ensure quote has more than 2 words
                q1 = quote_i
                
                # New metric: Compute fuzz score against quotes extracted from reference counterspeech
                if ref_extracted_quotes:  # Check if reference has quotes
                    ref_cs_fz_scores = [fuzz.partial_ratio(ref_q, quote_i) for ref_q in ref_extracted_quotes
                                        if len(ref_q.split()) > 2]
                    max_ref_cs_score = max(ref_cs_fz_scores) if ref_cs_fz_scores else 0
                    ref_cs_score += max_ref_cs_score
                else:
                    ref_cs_score += 0  # No quotes in reference, score 0

                count_q += 1

        extracted_quotes_list.append(q1)
        if count_q > 0:
            ref_cs_score /= count_q
            total_ref_cs_score += ref_cs_score
            total_valid_cs += 1
        else:
            ref_cs_score = None

        data.append([cs, q1, ref_cs_score])

    # Create DataFrame with new metric
    df = pd.DataFrame(data, columns=['Counterspeech', 'Quote', 'Ref_CS_Quote_Fuzz'])
    df.rename(columns={'Ref_CS_Quote_Fuzz': 'QC'}, inplace=True)
    print("Mean QC:", df['QC'].mean())
    df.to_csv('QC.csv', index=False)

    try:
        ref_cs_quote_fuzz = total_ref_cs_score / total_valid_cs if total_valid_cs > 0 else 0
    except Exception as e:
        print("Exception during metric calculation:", e)
        ref_cs_quote_fuzz = None

    return ref_cs_quote_fuzz

def calculate_quote_fuzz_score(gen_cs, ref_cs):
    """Calculate fuzz score between quotes extracted from generated and reference counterspeech"""
    try:
        # Clean inputs
        gen_cs = clean(gen_cs)
        ref_cs = clean(ref_cs)

        # Extract quotes (text within double quotes)
        gen_quotes = [q for q in re.findall(r'"(.*?)"', gen_cs) if len(q.split()) > 2]
        ref_quotes = [q for q in re.findall(r'"(.*?)"', ref_cs) if len(q.split()) > 2]

        # Return 0 if no valid quotes found
        if not gen_quotes or not ref_quotes:
            logging.warning(f"No valid quotes found: gen_cs='{gen_cs[:50]}...', ref_cs='{ref_cs[:50]}...'")
            return 0

        # Calculate fuzz scores for all quote pairs
        fuzz_scores = []
        for gen_quote in gen_quotes:
            for ref_quote in ref_quotes:
                score = fuzz.partial_ratio(gen_quote, ref_quote)
                fuzz_scores.append(score)

        # Return maximum score (or average if preferred)
        return max(fuzz_scores) if fuzz_scores else 0

    except Exception as e:
        logging.error(f"Error in calculate_quote_fuzz_score: {e}")
        return 0
    
def compute_metrics(data, batch_size=32, finetune=False):
    """Compute all metrics for the dataset"""
    metrics = []
    smoother = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Initialize models
    try:
        model = SentenceTransformer("/home/models/sentence-transformers-all-mpnet-base-v2")
        toxicity_model = Detoxify('unbiased')
    except Exception as e:
        print(f"Warning: Failed to load models: {e}")
        return []
    
    for idx, row in data.iterrows():
        try:
            reference = str(row['Reference CS'])
            hypothesis = str(row['CS'])

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

            try:
            
                with torch.no_grad():
                    P, R, F1 = bert_score([hypothesis], [reference], 
                                        lang='en',
                                        model_type='roberta-large',
                                        num_layers=9,
                                        )
                    # Apply penalty for short outputs
                    bert_f1 = F1.mean().item() * length_penalty
            except:
                bert_f1 = 0


            # Calculate toxicity
            toxicity = calculate_toxicity(hypothesis, toxicity_model)
            # toxicity = 0
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
    ref_cs = data['Reference CS']
    
    counterspeech = counterspeech.apply(clean)
    style_list = list(style)
    cs_list = list(counterspeech)
    
    style_score = nsp_score =  quote_avg_fuzz = quote_presence = None
    
    cleaned_cs = []
    import re
    for cs in counterspeech:
        cleaned_text = re.sub(r'"[^"]*"', '', cs)
        cleaned_text = cleaned_text.lower()
        cleaned_cs.append(cleaned_text)
    
    
    labels = list(data['Style'].map({"Gandhi":0, "Mandela":1}))
    
    style_score, style_score_2 = calculate_category_accuracy(cleaned_cs, labels)
    
    
    qc = fuzzy_match(cs_list, style_list, ref_cs)
    
    return (style_score, qc)
    
    
def main():
    parser = argparse.ArgumentParser(description="Compute metrics for text generation evaluation")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--fine_tune", type=int, default=1, help="fine tuned model output?")

    args = parser.parse_args()
    # Download required NLTK data
    download_nltk_data()
    # Read data
    try:
        data = pd.read_csv(args.input_file, encoding="ISO-8859-1")
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    # Get ground truth data
    try:
        eval_df = pd.read_csv('/home/aswini/Forth_project_on_Person_inspired_CS/baselines/input/Test_new.csv')
        eval_df = eval_df.head(len(data))
        data['Reference CS'] = list(eval_df['Counterspeech'])
        data['Hatespeech'] = list(eval_df['Hatespeech'])
        data['Style'] = list(eval_df['Style'])
        data['Target'] = list(eval_df['Target'])
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    data['Reference CS'] = data['Reference CS'].str.lower()
    data['CS'] = data['CS'].astype(str).str.lower()
    
    print("Input data Shape", data.shape)
    print("Check for null\n Before \n", data.isnull().sum())
    
    data = data[~(data['CS'].isnull())]
    
    data = data.reset_index()
    
    # Compute metrics
    print("Computing metrics...")
    # Traditional metrics
    metrics = compute_metrics(data, args.batch_size, finetune= args.fine_tune)
    metrics = None
    # SC, QC metrics
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
    
    # Save results to CSV
    try:
        results_df = pd.DataFrame(metrics, columns=headers)
        results_df['Hatespeech'] = data['Hatespeech'].values
        results_df['Counterspeech'] = data['CS'].values
        results_df['Reference Counterspeech'] = data['Reference CS'].values
        results_df = results_df[[
                                'Hatespeech', 'Counterspeech', 'Reference Counterspeech'
                            ] + headers]
        output_file = args.input_file.rsplit('.', 1)[0] + '_metrics.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    main()