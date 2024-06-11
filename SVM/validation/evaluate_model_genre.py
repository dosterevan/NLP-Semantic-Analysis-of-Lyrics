import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from scipy import sparse
import joblib

def load_data(tfidf_matrix_file, labels_file):
    tfidf_matrix = sparse.load_npz(tfidf_matrix_file)
    labels_df = pd.read_csv(labels_file)
    labels = labels_df['genre']
    return tfidf_matrix, labels

def load_model(model_file):
    svm_model = joblib.load(model_file)
    return svm_model

def evaluate_model(svm_model, X_test, y_test):
    y_pred = svm_model.predict(X_test)
    
    # Ensure consistent data types
    y_test = y_test.astype(str)
    y_pred = pd.Series(y_pred).astype(str)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    return accuracy

def main():
    combined_features_file = "SVM/train/combined_features.npz"
    labels_file = "SVM/train/labels.csv"
    model_file = "SVM/train/svm_model.pkl"
    combined_features, labels = load_data(combined_features_file, labels_file)
    svm_model = load_model(model_file)
    evaluation_file = "SVM/evaluation.txt"
    # try:
    #     with open(evaluation_file, 'r') as file:
    #         evaluation = float(file.read())
    #         print("Evaluation already performed. Accuracy:", evaluation)
    # except FileNotFoundError:
    X_train, X_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.2, random_state=42)
    evaluation = evaluate_model(svm_model, X_test, y_test)
    with open(evaluation_file, 'w') as file:
        file.write(str(evaluation))
        print("Evaluation saved to", evaluation_file)

if __name__ == "__main__":
    main()
