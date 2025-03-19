"""
python script.py --csv_path path/to/style_test_data.csv \
                 --model_path /home/models/roberta-large \
                 --tokenizer_path /home/models/roberta-large \
                 --checkpoint_path /home/aswini/Forth_project_on_Person_inspired_CS/author_style_classifier/saved_models/model_gandhi_style_epoch_2_fold9.pth
"""
import argparse
import pandas as pd
import numpy as np
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score
import torch
from torch.nn import functional as F

def calculate_category_accuracy(sentences, labels, model_path, tokenizer_path, checkpoint_path, device):
    roberta_tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    
    roberta_input_ids = []
    roberta_attention_masks = []
    sentence_ids = []
    counter = 0

    for sent in sentences:
        roberta_encoded_dict = roberta_tokenizer.encode_plus(
            sent, add_special_tokens=True, max_length=150, pad_to_max_length=True,
            return_attention_mask=True, return_tensors='pt'
        )
        roberta_input_ids.append(roberta_encoded_dict['input_ids'])
        roberta_attention_masks.append(roberta_encoded_dict['attention_mask'])
        sentence_ids.append(counter)
        counter += 1
    
    
    roberta_input_ids = torch.cat(roberta_input_ids, dim=0)
    roberta_attention_masks = torch.cat(roberta_attention_masks, dim=0)
    labels = torch.tensor(labels)
    sentence_ids = torch.tensor(sentence_ids)

    dataset = TensorDataset(roberta_input_ids, roberta_attention_masks, labels)
    dataloader = DataLoader(dataset, batch_size=32)
    
    roberta_model = RobertaForSequenceClassification.from_pretrained(
        model_path, num_labels=2, output_attentions=False, output_hidden_states=False
    )
    roberta_model.to(device)
    roberta_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    roberta_model.eval()

    total_eval_accuracy = 0
    total_f1_score = 0
    
    for batch in dataloader:
        b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]
        with torch.no_grad():
            loss, logits = roberta_model(b_input_ids, attention_mask=b_input_mask, labels=b_labels, return_dict=False)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        total_f1_score += f1_value(logits, label_ids)

    avg_accuracy = total_eval_accuracy / len(dataloader)
    avg_f1_score = total_f1_score / len(dataloader)
    return avg_f1_score, avg_accuracy

def flat_accuracy(preds, labels):
    return np.mean(np.argmax(preds, axis=1) == labels)

def f1_value(preds, labels):
    return f1_score(labels, np.argmax(preds, axis=1), average='weighted')

def predict_using_threshold(outputs, threshold=0.5):
    test_predictions = []
    test_prob = []
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    # predicted_prob = torch.max(probs, dim=1).values
    predicted_prob = probs[:, 1]
#     print(predicted_prob)
    for p in predicted_prob:
        if(p>=threshold):
            test_predictions.append(1)
        else:
            test_predictions.append(0)
            # test_prob.append(predicted_prob)
    return test_predictions
        
def predict(model, tokenizer, x_filter):
    tokenized_texts = tokenizer(x_filter, padding=True, truncation=True, return_tensors='pt')
    inputs = tokenized_texts['input_ids']
    masks = tokenized_texts['attention_mask']
    data = TensorDataset(inputs, masks)
    batch_size = 8
    dataloader = DataLoader(data, sampler=RandomSampler(data), batch_size=batch_size)
    test_predictions = []
    test_prob = []

    device = 'cuda'
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
        with torch.no_grad():
            outputs = model(**inputs)
        test_predictions.extend(predict_using_threshold(outputs, 0.1))
    
    return test_predictions

def calculate_text_style_accuracy(sentences):
    import torch
    device='cuda'
    model = RobertaForSequenceClassification.from_pretrained("/home/models/roberta-large", num_labels=2)
    model.load_state_dict(torch.load('/home/aswini/Forth_project_on_Person_inspired_CS/gandhi-evaluation/saved_models/model_gandhi_style_epoch_3_fold0.pth', map_location=device))
    model.to(device)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    tokenized_texts = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    inputs = tokenized_texts['input_ids']
    masks = tokenized_texts['attention_mask']

    data = TensorDataset(inputs, masks)
    
    batch_size = 8
    dataloader = DataLoader(data, sampler=RandomSampler(data), batch_size=batch_size)

    import torch
    import numpy as np
    from sklearn.metrics import f1_score
    test_predictions = []
    test_predictions2 = []
    test_true_labels = []
    prob = []

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        # test_predictions2.extend(predict_using_threshold(outputs, 0.5))
        # logits = outputs.logits
        # probs = F.softmax(logits, dim=-1)
        # predicted_prob = torch.max(probs, dim=1).values
        predicted_prob = probs[:, 1]
        prob.extend(predicted_prob.detach().cpu().tolist())

    # print(prob)
    print("Gandhi :", np.sum(prob)/len(prob))

    # print("# Gandhi(threshold = 0.8) : ", np.sum(test_predictions2)/len(test_predictions2))  

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--csv_path', type=str, required=True, help='Path to the test CSV file')
    # parser.add_argument('--model_path', type=str, required=True, help='Path to the RoBERTa model')
    # parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the RoBERTa tokenizer')
    # parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint')
    # args = parser.parse_args()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # df_test = pd.read_csv(args.csv_path, encoding='ISO-8859-1')
    # df_test['Labels'] = df_test['Style'].map({"Gandhi": 0, "Mandela": 1})
    
    # f1, acc = calculate_category_accuracy(df_test['CS'], df_test['Labels'], args.model_path, args.tokenizer_path, args.checkpoint_path, device)
    # print(f"F1 Score: {f1:.2f}")
    # print(f"Accuracy: {acc:.2f}")

    files = [
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/one-shot/t5.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/one-shot/bart.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/one-shot/gpt.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/one-shot/llama.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/one-shot/mistral.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/few-shot/t5.csv',
        # # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/one-shot/bart.csv',
        # # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/few-shot/gpt.csv',
        '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/one-shot/t5.csv',
        '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/inference_scripts/results/zero-shot/llama.csv',
        '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/inference_scripts/results/zero-shot/mistral.csv',
        '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/few-shot/t5.csv',
        '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/inference_scripts/results/few-shot/llama.csv',
        '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/inference_scripts/results/few-shot/mistral.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/inference_scripts/results/zero-shot/llama.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/inference_scripts/results/zero-shot/mistral.csv'
        '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/finetune/t5.csv',
        '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/finetune/llama.csv',
        '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/finetune/mistral.csv'


    ]

    DF = pd.read_csv('/home/aswini/Forth_project_on_Person_inspired_CS/baselines/input/Person_specific_counterspeech3.csv')
    for file in files:
        print(file)
        df = pd.read_csv(file)
        gandhi_indices = DF['Style']=='Gandhi'
        sentences = df[gandhi_indices]['CS']
        sentences = list(sentences)
        calculate_text_style_accuracy(sentences)

if __name__ == '__main__':
    main()