import os
import re

def load_custom_stopwords(stopwords_dir):
    """Load stopwords from all text files in the specified directory."""
    custom_stopwords = set()
    
    if os.path.exists(stopwords_dir):
        for file_name in os.listdir(stopwords_dir):
            if file_name.endswith('.txt'):
                file_path = os.path.join(stopwords_dir, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    words = [line.strip() for line in f if line.strip()]
                    custom_stopwords.update(words)
                print(f"Loaded stopwords from: {file_name}")
    else:
        print(f"Warning: Stopwords directory '{stopwords_dir}' not found!")
    
    return custom_stopwords

def remove_stopwords(text, custom_stopwords):
    words = text.split()
    return ' '.join([word for word in words if word not in custom_stopwords])

def remove_timestamps(text):
    pattern = r'\d+\.\d+:\s*'
    return re.sub(pattern, '', text).strip()

def remove_single_letters(text):
    return re.sub(r'\b[ء-ي]\b', '', text)

def remove_repeated_words(text):
    return re.sub(r'(\b\w+\b)(\s+\1\b)+', r'\1', text)

def spell_check(text):
    for incorrect, correct in SPELL_CORRECTIONS.items():
        text = text.replace(incorrect, correct)
    return text

def clean_transcript(text, custom_stopwords):
    text = remove_timestamps(text)
    text = remove_stopwords(text, custom_stopwords)
    # text = remove_single_letters(text)
    # text = remove_repeated_words(text)
    # text = spell_check(text)
    return text

def clean_transcripts(raw_dir, clean_dir, stopwords_dir):
    if not os.path.exists(clean_dir):
        os.makedirs(clean_dir)
    
    # Load custom stopwords from files
    custom_stopwords = load_custom_stopwords(stopwords_dir)
    print(f"Loaded {len(custom_stopwords)} custom stopwords")
    
    for file_name in os.listdir(raw_dir):
        if file_name.endswith('.txt'):
            raw_file_path = os.path.join(raw_dir, file_name)
            with open(raw_file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            cleaned_text = clean_transcript(raw_text, custom_stopwords)
            
            cleaned_file_path = os.path.join(clean_dir, file_name)
            with open(cleaned_file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            print(f"Saved cleaned transcript as {cleaned_file_path}")

# Keep this dictionary for spell check functionality
SPELL_CORRECTIONS = {
    # Add corrections here
}

if __name__ == "__main__":
    input_dir = 'Raw Data'
    output_dir = 'Clean Data'
    stopwords_dir = 'stopwords'  # Directory containing the 4 stopword files
    clean_transcripts(input_dir, output_dir, stopwords_dir)
