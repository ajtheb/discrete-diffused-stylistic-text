import re
from wordcloud import STOPWORDS, WordCloud
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from collections import Counter
from itertools import islice
from nltk.util import ngrams


def get_top_ngrams(text, n, top_k=10):
    words = text.split()
    n_grams = list(ngrams(words, n))
    n_gram_freq = Counter(n_grams)
    return n_gram_freq.most_common(top_k)


def plot_ngrams(ngrams_freq, title):
    labels, values = zip(*ngrams_freq)
    labels = [' '.join(label) for label in labels]
    plt.figure(figsize=(10, 5))
    plt.barh(labels, values, color='skyblue')
    plt.xlabel("Frequency")
    plt.ylabel("N-gram")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{title}_wordcloud.png", dpi=300)

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
    
    text = re.sub(r'"[^"]*"', '', text)
    
    text = text.replace("dear", "")
    text = text.replace("friend", "")
    text = text.replace("let", "")
    text = text.replace("us", "")

    return text

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Word cloud and N-gram frequency of counterspeeches")
    parser.add_argument("--leader", type=str, required=True, help="Gandhi/Mandela")
    
    args = parser.parse_args()
    leader = args.leader
    
    df = pd.read_csv('Person_specific_counterspeech3.csv')
    cs = df[df['Style'] == leader]['Counterspeech']
    cs = cs.apply(clean)
    
    text = ' '.join(cs.astype(str).tolist())
    
    text = text.lower()
    
    stopwords = set(STOPWORDS)
    text = ' '.join(word for word in text.split() if word not in stopwords)
    
    text = re.sub(r"[^\w\s]", "", text)
    
    # Generate Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"{leader} Counterspeech Word Cloud")
    plt.show()
    plt.savefig(f"{leader}_wordcloud.png", dpi=300)
    
    # Generate N-gram counts and plots
    for n in [1,2, 3,4]:  # Bigrams and Trigrams
        top_ngrams = get_top_ngrams(text, n, top_k=10)
        print(top_ngrams)
        plot_ngrams(top_ngrams, title=f"Top {n}-grams in {leader} Counterspeech")
    