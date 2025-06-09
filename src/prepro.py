import pandas as pd
import re
import string
import nltk
import os

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Inisialisasi Stemmer dan Stopwords
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))
whitelist = {"bajingan"}

# Load kamus alay
current_dir = os.path.dirname(os.path.abspath(__file__))
kamus_path = os.path.join(current_dir, '..', 'data', 'new_kamusalay.csv')
kamus_df = pd.read_csv(kamus_path, header=None, names=['slang', 'formal'], encoding='ISO-8859-1')
slang_dict = dict(zip(kamus_df['slang'], kamus_df['formal']))

def case_folding(text):
    return text.lower()

def clean_data(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+([._]\w+)*|\busername\b', '', text)
    text = re.sub(r'(.)\1{1,}', r'\1', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'[\t\n\r ]+', ' ', text).strip()
    return text

def tokenization(text):
    return word_tokenize(text)

def normalization(tokens):
    return [slang_dict.get(word, word) for word in tokens]

def filtering(tokens):
    return [word for word in tokens if word not in stop_words and word not in string.punctuation and len(word) > 1]

def stemming(tokens):
    return [token if token in whitelist else stemmer.stem(token) for token in tokens]

def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == '':
        return ''
    
    print("\n[Original]:", text)
    text = case_folding(text)
    print("[Case Folding]:", text)
    text = clean_data(text)
    print("[Cleaned Text]:", text)
    tokens = tokenization(text)
    print("[Tokenized]:", tokens)
    tokens = normalization(tokens)
    print("[Normalized]:", tokens)
    tokens = filtering(tokens)
    print("[Filtered]:", tokens)
    tokens = stemming(tokens)
    print("[Stemmed]:", tokens)

    return ' '.join(tokens)

def preprocess_and_save_pipeline(file_path):
    print("Mulai preprocessing data...")

    data = pd.read_csv(file_path)
    if 'Komentar' not in data.columns:
        raise ValueError("Kolom 'Komentar' tidak ditemukan di dalam dataset.")
    data.dropna(subset=['Komentar'], inplace=True)

    # Terapkan preprocessing
    data['Komentar'] = data['Komentar'].apply(preprocess_text)

    processed_data_path = os.path.join(current_dir, '..', 'data', 'processed_comments.csv')
    data.to_csv(processed_data_path, index=False)

    print(f"\nHasil preprocessing disimpan di: {processed_data_path}")

if __name__ == "__main__":
    file_path1 = os.path.join(current_dir, '..', 'data', 'cyberbullyingnew2.csv')
    file_path2 = os.path.join(current_dir, '..', 'data', 'cyberbullyingnew.csv')

    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)

    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df.drop_duplicates(subset=['Komentar'], inplace=True)

    combined_file_path = os.path.join(current_dir, '..', 'data', 'combined_cyberbullying.csv')
    combined_df.to_csv(combined_file_path, index=False)

    preprocess_and_save_pipeline(combined_file_path)
