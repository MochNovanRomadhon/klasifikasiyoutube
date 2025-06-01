import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Mengabaikan peringatan FutureWarning

import os
import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Fungsi untuk melatih model SVM
def train_svm(X_train, y_train):
    print("Melatih model SVM...")
    model = SVC(kernel='linear')  # Menggunakan kernel linear
    model.fit(X_train, y_train)   # Melatih model dengan data latih
    return model

# Fungsi untuk mengevaluasi model dan menyimpan hasil prediksi
def evaluate_model(model, X_test, y_test, original_df_test, output_path):
    print("\nMengevaluasi model...")
    y_pred = model.predict(X_test)  # Melakukan prediksi pada data uji
    cm = confusion_matrix(y_test, y_pred)  # Membuat confusion matrix
    TN, FP, FN, TP = cm.ravel()  # Mengambil nilai-nilai dari confusion matrix

    # Menghitung metrik evaluasi
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)

    # Menampilkan hasil evaluasi
    print("\nConfusion Matrix:")
    print(cm)
    print(f"Akurasi   : {accuracy:.3f}")
    print(f"Precision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")
    print(f"F1 Score  : {f1:.3f}")

    # Membuat DataFrame untuk melihat hasil prediksi
    pred_df = pd.DataFrame({
        'text': original_df_test['Komentar'],        # Kolom teks komentar
        'actual_label': y_test,                      # Label asli
        'predicted_label': y_pred                    # Label hasil prediksi
    })

    print("\nContoh Prediksi:")
    print(pred_df.head(10))  # Menampilkan 10 hasil prediksi pertama

    # Menyimpan hasil prediksi ke file CSV
    pred_df.to_csv(output_path, index=False)
    print(f"\n✅ Hasil prediksi disimpan ke {output_path}")

# Fungsi untuk menyimpan model yang sudah dilatih
def save_model(model, file_path):
    print("\nMenyimpan model...")
    joblib.dump(model, file_path)  # Simpan model ke file dengan joblib
    print(f"✅ Model disimpan di {file_path}")

# Eksekusi utama program
if __name__ == "__main__":
    print("Memulai proses pemodelan...\n")

    # Menentukan path file yang dibutuhkan
    current_dir = os.path.dirname(__file__)  # Direktori saat ini
    tfidf_data_path = os.path.join(current_dir, '..', 'data', 'processed_tfidf.pkl')
    model_path = os.path.join(current_dir, '..', 'models', 'svm_model.pkl')
    original_data_path = os.path.join(current_dir, '..', 'data', 'combined_cyberbullying.csv')
    prediction_output_path = os.path.join(current_dir, '..', 'data', 'svm_predictions.csv')

    # Memuat data TF-IDF dan label dari file .pkl
    X, y = joblib.load(tfidf_data_path)

    # Membaca file data asli dan menyelaraskan indeks dengan label
    original_df = pd.read_csv(original_data_path)
    original_df = original_df.loc[y.index].reset_index(drop=True)
    y = y.reset_index(drop=True)
    original_df = original_df.reset_index(drop=True)

    # Membagi data menjadi data latih dan uji (stratifikasi berdasarkan label)
    train_idx, test_idx = train_test_split(y.index, test_size=0.2, stratify=y, random_state=42)

    # Ambil data latih dan uji berdasarkan indeks yang dibagi
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Ambil data komentar asli yang sesuai untuk evaluasi hasil prediksi
    original_df_test = original_df.iloc[test_idx].reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Menampilkan jumlah data latih dan uji
    print(f"Jumlah data latih: {X_train.shape[0]}")
    print(f"Jumlah data uji  : {X_test.shape[0]}\n")

    # Melatih model SVM
    model = train_svm(X_train, y_train)

    # Mengevaluasi model dan menyimpan hasil prediksi
    evaluate_model(model, X_test, y_test, original_df_test, prediction_output_path)

    # Menyimpan model ke file
    save_model(model, model_path)

    print("Proses selesai!")