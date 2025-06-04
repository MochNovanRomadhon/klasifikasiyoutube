import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_data(data):
    data.dropna(subset=['Komentar'], inplace=True)
    return data

def apply_tfidf(data):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.9, sublinear_tf=True)
    X = tfidf_vectorizer.fit_transform(data['Komentar'])
    y = data['Kategori'].reset_index(drop=True)
    return X, y, tfidf_vectorizer

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    input_file_path = os.path.join(current_dir, '..', 'data', 'processed_comments.csv')
    tfidf_model_path = os.path.join(current_dir, '..', 'models', 'tfidf_vectorizer.pkl')
    tfidf_data_path = os.path.join(current_dir, '..', 'data', 'processed_tfidf.pkl')

    data = load_data(input_file_path)
    data = clean_data(data)
    X, y, tfidf_vectorizer = apply_tfidf(data)

    joblib.dump(tfidf_vectorizer, tfidf_model_path)
    joblib.dump((X, y), tfidf_data_path)

    print(f"✅ Model TF-IDF disimpan di {tfidf_model_path}")
    print(f"✅ Data TF-IDF (sparse) disimpan di {tfidf_data_path}")
