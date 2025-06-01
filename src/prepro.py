



# Import library yang dibutuhkan
import pandas as pd
import re
import string
import nltk
import os
import joblib

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Inisialisasi stemmer Bahasa Indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Mendefinisikan path file kamus alay (slang)
current_dir = os.path.dirname(os.path.abspath(__file__))
kamus_path = os.path.join(current_dir, '..', 'data', 'new_kamusalay.csv')

# Load kamus alay (slang -> formal) ke dictionary
kamus_df = pd.read_csv(kamus_path, header=None, names=['slang', 'formal'], encoding='ISO-8859-1')
slang_dict = dict(zip(kamus_df['slang'], kamus_df['formal']))

# Kelas untuk preprocessing teks
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('indonesian'))  # Stopwords Bahasa Indonesia
        self.whitelist = {"bajingan"}  # Whitelist kata kasar yang tetap dipertahankan

    def case_folding(self, text):
        return text.lower()

    def clean_text(self, text):
        # Bersihkan URL, mention, huruf berulang, simbol, kata pendek, dll.
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+([._]\w+)*|\busername\b', '', text)
        text = re.sub(r'(.)\1{1,}', r'\1', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\b\w{1,2}\b', '', text)
        text = re.sub(r'[\t\n\r ]+', ' ', text).strip()
        return text

    def tokenize(self, text):
        return word_tokenize(text)

    def normalize_with_dictionary(self, tokens):
        return [slang_dict.get(word, word) for word in tokens]

    def filter_tokens(self, tokens):
        return [word for word in tokens if word not in self.stop_words and word not in string.punctuation and len(word) > 1]

    def stemming(self, tokens):
        return [token if token in self.whitelist else stemmer.stem(token) for token in tokens]

    def preprocess_text(self, text):
        if not isinstance(text, str) or text.strip() == '':
            return ''
        # Proses berurutan: Case folding -> cleaning -> tokenizing -> normalisasi -> filter -> stemming
        print("\n[Original]:", text)
        text = self.case_folding(text)
        print("[Case Folding]:", text)
        text = self.clean_text(text)
        print("[Cleaned Text]:", text)

        tokens = self.tokenize(text)
        print("[Tokenized]:", tokens)
        tokens = self.normalize_with_dictionary(tokens)
        print("[Normalized (slang to formal)]:", tokens)
        tokens = self.filter_tokens(tokens)
        print("[Filtered Tokens]:", tokens)
        tokens = self.stemming(tokens)
        print("[Stemmed]:", tokens)

        return ' '.join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.Series):
            return X.apply(self.preprocess_text)
        elif isinstance(X, list):
            return [self.preprocess_text(text) for text in X]
        else:
            return X

# Fungsi untuk preprocessing seluruh data dan menyimpannya
def preprocess_and_save_pipeline(file_path):
    print("Mulai preprocessing data...")

    # Load dataset
    data = pd.read_csv(file_path)
    if 'Komentar' not in data.columns:
        raise ValueError("Kolom 'Komentar' tidak ditemukan di dalam dataset.")

    # Hapus kolom Unnamed jika ada dan hilangkan data kosong
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    data.dropna(subset=['Komentar'], inplace=True)

    # Buat pipeline dengan class custom TextPreprocessor
    pipeline = Pipeline([
        ('text_preprocessor', TextPreprocessor())
    ])

    # Terapkan preprocessing ke kolom 'Komentar'
    data['Komentar'] = pipeline.fit_transform(data['Komentar'])

    # Simpan data dan pipeline
    processed_data_path = os.path.join(current_dir, '..', 'data', 'processed_comments.csv')
    pipeline_path = os.path.join(current_dir, '..', 'models', 'preprocess_pipeline.pkl')

    data.to_csv(processed_data_path, index=False)
    joblib.dump(pipeline, pipeline_path)

    print(f"\nHasil preprocessing disimpan di: {processed_data_path}")
    print(f"Pipeline disimpan di: {pipeline_path}")

# Program utama: Gabungkan dua file CSV, hilangkan duplikat, lalu lakukan preprocessing
if __name__ == "__main__":
    file_path1 = os.path.join(current_dir, '..', 'data', 'cyberbullyingnew2.csv')
    file_path2 = os.path.join(current_dir, '..', 'data', 'cyberbullyingnew3.csv')

    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)

    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df.drop_duplicates(subset=['Komentar'], inplace=True)

    combined_file_path = os.path.join(current_dir, '..', 'data', 'combined_cyberbullying.csv')
    combined_df.to_csv(combined_file_path, index=False)

    preprocess_and_save_pipeline(combined_file_path)