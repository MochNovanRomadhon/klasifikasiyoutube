import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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

def evaluate_model(model, X_test, y_test, original_df_test):
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
    tfidf_data_path = os.path.join(current_dir, '..', 'data', 'processed_tfidf.pkl')
    model_path = os.path.join(current_dir, '..', 'models', 'svm_model.pkl')
    original_data_path = os.path.join(current_dir, '..', 'data', 'combined_cyberbullying.csv')

    X, y = joblib.load(tfidf_data_path)

    original_df = pd.read_csv(original_data_path)
    original_df = original_df.loc[y.index].reset_index(drop=True)
    y = y.reset_index(drop=True)
    original_df = original_df.reset_index(drop=True)

    train_idx, test_idx = train_test_split(y.index, test_size=0.2, stratify=y, random_state=42)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    original_df_test = original_df.iloc[test_idx].reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    model = train_svm(X_train, y_train)
    evaluate_model(model, X_test, y_test, original_df_test)
    save_model(model, model_path)

    print("Proses selesai!")