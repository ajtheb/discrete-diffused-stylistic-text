import pandas as pd
import re
from rapidfuzz import fuzz, process
import time
import json
import argparse
from bert_score import score  # requires: pip install bert-score
import warnings 

from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
    

warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")



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
3. Quote recall: how many quotes are covered wrt to the dataset of quotes?
    Range = [0-1]
4. Quote Exact match: how many quotes are exactly matched?
    Range = [0-1]
5. Quote Semantic score: Average similarity between extracted quote and hatespeech using BERT Score.
    Range = [0-1]
"""
def fuzzy_match(mandela_quotes, gandhi_quotes, counterspeeches, hatespeech, threshold=80):
    data = []
    total_score = 0
    total_valid_cs = 0
    present_quotes = set()

    # For Exact Match Accuracy metric
    total_extracted = 0
    exact_match_count = 0

    # For BERTScore metric between extracted quote and hatespeech
    total_quote_bertscore = 0
    count_quote_bertscore = 0

    extracted_quotes_list = []

    for index, cs in enumerate(counterspeeches):
        # Alternate between the two authors based on index:
        if index % 2:
            quotes = mandela_quotes
        else:
            quotes = gandhi_quotes

        extracted_quotes = re.findall(r'"(.*?)"', cs)
        score_cs = 0
        count_q = 0

        # Get the corresponding hate speech text.
        # Assuming hatespeech is a pandas Series.
        current_hate = hatespeech.iloc[index] if hasattr(hatespeech, "iloc") else hatespeech[index]

        # if(len(extracted_quotes)>0):
        #     extracted_quotes_list.append(extracted_quotes[0])
        # else:
        #     extracted_quotes_list.append("")

        q1 = "Quote"
        for quote_i in extracted_quotes:
            if len(quote_i) > 2:
                total_extracted += 1
                q1 = quote_i
                # Check for an exact match in the reference quotes list.
                if quote_i in quotes:
                    exact_match_count += 1

                fz_scores = [fuzz.partial_ratio(q, quote_i) for q in quotes]
                max_fuzz_score = max(fz_scores)
                original_quote_index = fz_scores.index(max_fuzz_score)
                present_quotes.add(original_quote_index)
                score_cs += max_fuzz_score
                count_q += 1

                
                # total_quote_bertscore += bert_f1_score
                # count_quote_bertscore += 1

        extracted_quotes_list.append(q1)
        if count_q > 0:
            score_cs /= count_q
            total_score += score_cs
            total_valid_cs += 1
        else:
            score_cs = None
        
        data.append([cs, extracted_quotes, score_cs])

    # Compute BERTScore F1 between the extracted quote and the corresponding hatespeech
    # try:
    #     # bert_score returns tensors for precision, recall, and F1 scores.
    #     _, _, bert_f1 = score(extracted_quotes_list, list(hatespeech), lang="en", verbose=False)
    #     # Get scalar value from tensor
    #     bert_f1_score = bert_f1[0].item()
    # except Exception as e:
    #     print("BERTScore calculation error:", e)
    #     bert_f1_score = 0.0
    # # Create DataFrame with per-counterspeech details.
    df = pd.DataFrame(data, columns=['Counterspeech', 'Quote', 'Max_Fuzz_Score'])
    df.to_csv('cs_fuzzscore.csv', index=False)

    try:
        quote_avg_fuzz = total_score / len(counterspeeches)
        quote_presence = total_valid_cs / len(counterspeeches)
        quote_recall = len(list(present_quotes)) / (len(set(mandela_quotes)) + len(set(gandhi_quotes)))
        print("Number:", len(list(set(present_quotes))))
        quote_score = 0.7 * (quote_avg_fuzz/100) + 0.3 * quote_presence
        exact_match_accuracy = (exact_match_count / total_extracted) if total_extracted > 0 else 0
        # avg_quote_bertscore = bert_f1_score
    except Exception as e:
        print("Exception during metric calculation:", e)
        quote_avg_fuzz = quote_presence = quote_recall = quote_score = exact_match_accuracy = avg_quote_bertscore = None

    print("Quote Average Fuzz:", quote_avg_fuzz)
    print("Quote Presence Accuracy:", quote_presence)
    print("Quote Recall:", quote_recall)
    print("Quote Score:", quote_score)
    print("Exact Match Accuracy:", exact_match_accuracy)
    # print("Average Quote BERTScore:", avg_quote_bertscore)

    return df

def calculate_ideology_metric(counterspeeches):
    with open("/home/aswini/Forth_project_on_Person_inspired_CS/baselines/quotes/merged_quotes.json", "r", encoding="utf-8") as f:
            quotes_data = json.load(f)
    gandhi_quotes = quotes_data['authors']['Mahatma Gandhi']['quotes']
    gandhi_quotes = list(set(gandhi_quotes))
    model_name = "sentence-transformers/all-mpnet-base-v2"
    # model_name = '/kaggle/input/stance-detect/transformers/1/1'
    model_kwargs = {"device": "cuda:1"}

    all_documents = []
    for q in gandhi_quotes:
        text_splitter = CharacterTextSplitter(
            chunk_size= 400,
            chunk_overlap = 40,
            separator='', strip_whitespace=False
        )
        chunk_list = text_splitter.split_text(q)
        #     print(len(text))
        #     print(len(chunk_list))
        for chunk in chunk_list:
            doc = Document(page_content=chunk)
            all_documents.append(doc)

    embeddings_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    path = 'gandhi_quotes'
    vectordb = Chroma(embedding_function=embeddings_model, persist_directory=path)
    # vectordb.get()
    llm = HuggingFacePipeline()
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})  # Example setting
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=True
    )
    scores = []

    for cs in counterspeeches:
        text = cs
        docs_and_scores = qa.retriever.vectorstore.similarity_search_with_relevance_scores(text, k=10)
        unique_texts = set()  # To store unique retrieved documents
        top_3_docs = []  # To store the final top 3 results
        top_3_docs_scores = []  # To store the final top 3 results

        for doc, score in docs_and_scores:
            text = doc.page_content  # Extract the text content from the document
            if text not in unique_texts:
                unique_texts.add(text)
                top_3_docs.append((text, score))  # Store text along with score
                top_3_docs_scores.append(score)
            if len(top_3_docs) == 3:  # Stop once we have 3 unique results
                break

        score = sum(top_3_docs_scores)/len(top_3_docs)
        scores.append(score)
        # print("score ", score)
    # print(file_path)
    print("final score:", sum(scores)/len(scores))


def main():
    parser = argparse.ArgumentParser(description="Quote match metric")
    # parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--cs_column", type=str, default="CS", help="Column of Counterspeech")

    args = parser.parse_args()

    files = [
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/one-shot/t5.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/one-shot/bart.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/one-shot/gpt.csv',
        '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/one-shot/llama.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/one-shot/mistral.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/few-shot/t5.csv',
        # # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/one-shot/bart.csv',
        # # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/one-shot/gpt.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/few-shot/llama.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/few-shot/mistral.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/finetune/t5.csv',
        # # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/one-shot/bart.csv',
        # # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/one-shot/gpt.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/finetune/llama.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/finetune/mistral.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/finetune/t5.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/inference_scripts/results/finetune/llama_r_8.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/inference_scripts/results/finetune/llama_r_32.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/inference_scripts/results/finetune/mistral_r_8.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/inference_scripts/results/finetune/mistral_r_32.csv'
        
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/inference_scripts/results/finetune/mistral_r_16.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/finetune/coarl_5000.csv',
        # # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/inference_scripts/results/finetune/llama.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/results/few-shot/t5.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/inference_scripts/results/few-shot/llama.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/inference_scripts/results/few-shot/mistral.csv',
        # '/home/aswini/Forth_project_on_Person_inspired_CS/baselines/input/Person_specific_counterspeech3.csv'
            ]

    for file in files:
        # Load and clean counterspeeches.
        # df = pd.read_csv(args.input_file)
        print(f"Running for file: {file}")
        df = pd.read_csv(file)
        cs = df[args.cs_column].apply(clean)
        # style_labels = df['Style']
        # x = df['Counterspeech'].apply(clean)
        hs = df['hatespeech']

        # Load quotes data.
        with open("merged_quotes.json", "r", encoding="utf-8") as f:
            quotes_data = json.load(f)

        mandela_quotes = quotes_data['authors']['Nelson Mandela']['quotes']
        gandhi_quotes = quotes_data['authors']['Mahatma Gandhi']['quotes']
        print("Number of Gandhi quotes", len(set(gandhi_quotes)))
        print("Number of Mandela quotes", len(set(mandela_quotes)))

        start_time = time.time()
        result = fuzzy_match(mandela_quotes, gandhi_quotes, cs, hs, 80)
        end_time = time.time()

        print(f"Match Result: {result}")
        
        # calculate_ideology_metric(counterspeeches=x)

        print(f"Time Taken: {end_time - start_time:.6f} seconds")

        
if __name__ == '__main__':
    main()
