# Import library yang dibutuhkan
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Fungsi untuk memuat data dari file CSV
def load_data(file_path):
    return pd.read_csv(file_path)

# Fungsi untuk membersihkan data dari entri kosong di kolom 'Komentar'
def clean_data(data):
    data.dropna(subset=['Komentar'], inplace=True)
    return data

# Fungsi untuk melakukan transformasi TF-IDF
def apply_tfidf(data):
    # Inisialisasi TF-IDF Vectorizer dengan parameter:
    # - max_df=0.9: mengabaikan kata yang muncul di lebih dari 90% dokumen
    # - sublinear_tf=True: menggunakan sublinear term frequency scaling
    tfidf_vectorizer = TfidfVectorizer(max_df=0.9, sublinear_tf=True)
    
    # Transformasi teks ke representasi vektor TF-IDF
    X = tfidf_vectorizer.fit_transform(data['Komentar'])
    
    # Ambil label target dari kolom 'Kategori'
    y = data['Kategori'].reset_index(drop=True)
    
    return X, y, tfidf_vectorizer


if __name__ == "__main__":
    # Tentukan direktori saat ini dan path untuk input dan output
    current_dir = os.path.dirname(__file__)
    input_file_path = os.path.join(current_dir, '..', 'data', 'processed_comments.csv')
    tfidf_model_path = os.path.join(current_dir, '..', 'models', 'tfidf_vectorizer.pkl')
    tfidf_data_path = os.path.join(current_dir, '..', 'data', 'processed_tfidf.pkl')

    # Muat dan bersihkan data
    data = load_data(input_file_path)
    data = clean_data(data)

    # Terapkan TF-IDF dan dapatkan X (fitur) dan y (label)
    X, y, tfidf_vectorizer = apply_tfidf(data)

    # Simpan model TF-IDF dan hasil transformasi (X, y) menggunakan joblib
    joblib.dump(tfidf_vectorizer, tfidf_model_path)
    joblib.dump((X, y), tfidf_data_path)

    # Informasi berhasil disimpan
    print(f"✅ Model TF-IDF disimpan di {tfidf_model_path}")
    print(f"✅ Data TF-IDF (sparse) disimpan di {tfidf_data_path}")