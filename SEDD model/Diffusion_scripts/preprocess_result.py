import argparse
import pandas as pd
def preprocess_text(text):
    try:
        # Step 1: Remove special tokens ([CLS], [SEP], [PAD])
        special_tokens = ['[CLS]', '[SEP]', '[PAD]','<|endoftext|>']
        for token in special_tokens:
            text = text.replace(token, '')
        
        # Step 2: Split text into words and remove consecutive repeated words
        words = text.split()
        filtered_words = []
        prev_word = None
        for word in words:
            if word.lower() != prev_word:
                filtered_words.append(word)
                prev_word = word.lower()
        
        # Step 3: Join words back into a string
        cleaned_text = ' '.join(filtered_words)
        print(cleaned_text)
        # Step 4: Extract counterspeech (assuming format: Hatespeech Gandhi/Mandela Counterspeech)
        # Find the position of Gandhi or Mandela to split
        gandhi_pos = cleaned_text.lower().find('gandhi')
        mandela_pos = cleaned_text.lower().find('mandela')
        
        # Take the earliest valid position (non-negative)
        if gandhi_pos != -1 and mandela_pos != -1:
            counterspeech_start = min(gandhi_pos, mandela_pos)
        elif gandhi_pos != -1:
            counterspeech_start = gandhi_pos
        elif mandela_pos != -1:
            counterspeech_start = mandela_pos
        else:
            counterspeech_start = -1
        
        if counterspeech_start != -1:
            # Extract text after Gandhi/Mandela
            counterspeech = cleaned_text[counterspeech_start:].split(' ', 1)[1].strip()
            return counterspeech
        else:
            # If neither Gandhi nor Mandela is found, return cleaned text
            return cleaned_text
    except:
        return None
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess text file to extract counterspeech.')
    parser.add_argument('--file_path', type=str, help='Path to the input text file')
    parser.add_argument('--save_path', type=str, help='Path to the output text file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # x = """[CLS] faggots they are the problem. luckily they are few. [SEP] mandela [SEP] [SEP] my dear friend, i understand that your words stem from a place of misunderstanding"""
    # print(preprocess_text(x))
    # Read input file
    try:
        df = pd.read_csv(args.file_path)
        # print(df.shape)
        # df = df[~(df['CS'].isnull())]
        # print(df.shape)
        df['CS'] = df['CS'].apply(preprocess_text)
        # Process text and print result
        # counterspeech = preprocess_text(input_text)
        # print(counterspeech)
        df.to_csv(args.save_path)
        
    except FileNotFoundError:
        print(f"Error: File '{args.file_path}' not found.")
    except Exception as e:
        print(f"Error processing file: {str(e)}")