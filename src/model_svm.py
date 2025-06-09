import os
import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

def train_svm(X_train, y_train):
    print("Melatih model SVM...")
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    print("\nMengevaluasi model...")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)

    print("\nConfusion Matrix:")
    print(cm)
    print(f"Akurasi   : {accuracy:.3f}")
    print(f"Precision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")
    print(f"F1 Score  : {f1:.3f}")

def save_model(model, file_path):
    print("\nMenyimpan model...")
    joblib.dump(model, file_path)
    print(f"✅ Model disimpan di {file_path}")

if __name__ == "__main__":
    print("Memulai proses pemodelan...\n")

    current_dir = os.path.dirname(__file__)
    tfidf_csv_path = os.path.join(current_dir, '..', 'data', 'processed_tfidf.csv')
    original_data_path = os.path.join(current_dir, '..', 'data', 'combined_cyberbullying.csv')
    model_path = os.path.join(current_dir, '..', 'models', 'svm_model.pkl')

    # Load data
    X = pd.read_csv(tfidf_csv_path)
    original_df = pd.read_csv(original_data_path)

    if 'Kategori' not in original_df.columns:
        raise ValueError("Kolom 'Kategori' tidak ditemukan dalam data asli!")

    y = original_df['Kategori'].fillna('').reset_index(drop=True)
    original_df = original_df.reset_index(drop=True)
    X = X.reset_index(drop=True)

    # Split data
    X_train, X_test, y_train, y_test, original_df_train, original_df_test = train_test_split(
        X, y, original_df, test_size=0.2, stratify=y, random_state=42
    )

    # Train, evaluate, save
    model = train_svm(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, model_path)

    print("Proses selesai!")