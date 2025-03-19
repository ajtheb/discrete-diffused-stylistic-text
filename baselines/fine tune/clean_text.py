import pandas as pd
import re
import argparse

def clean_text(text):
    # Remove unwanted encoded characters
    text = re.sub(r'â€š\.|‚\.', '', text)

    text = re.sub(r'‚\s*\.', '', text)  # Remove '‚ .' patterns
    text = re.sub(r'‚', '', text)

    # Remove repeated occurrences of " . " or similar patterns
    text = re.sub(r'(\s*\.\s*){2,}', '.', text)
    
    # Remove repeated occurrences of encoded characters
    text = re.sub(r'(‚\.\s*)+', '', text)
    
    # Remove hashtags and words following them
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'# \S+', '', text)
    
    # Remove text after "Audio:"
    text = re.sub(r'Audio:.*', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'Image:.*', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'Audio description of the image:.*', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'Audio track:.*', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'Music:.*', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'Audio description:.*', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'Audio message:.*', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'Audio description.*', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'Audio message.*', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'Source:*', '', text, flags=re.DOTALL).strip()
    
    
    
    # Normalize spaces and line breaks
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Ensure proper punctuation spacing
    text = re.sub(r'(\w)([.,!?])', r'\1 \2', text)
    
    return text



def main():
    parser = argparse.ArgumentParser(description="Clean text in a CSV file.")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("output_file", help="Path to save the cleaned CSV file")
    
    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.input_file)

    # Apply cleaning function
    df['CS'] = df['CS'].apply(clean_text)

    # Save cleaned CSV
    df.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    main()
