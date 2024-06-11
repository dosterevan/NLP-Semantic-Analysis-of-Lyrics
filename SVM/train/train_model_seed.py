import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from scipy import sparse
import joblib

def load_data(features_file, labels_file):
    # Load combined features matrix
    combined_features = sparse.load_npz(features_file)
    
    # Load labels
    labels_df = pd.read_csv(labels_file)
    labels = labels_df['seeds']
    
    # Check for and handle NaN values in labels
    if labels.isna().any():
        print("Found NaN values in labels, removing them...")
        labels = labels.dropna()
        combined_features = combined_features[labels.index]
    
    return combined_features, labels

def train_svm(combined_features, labels):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.2, random_state=42)
    
    # Initialize SVM classifier
    svm_classifier = SVC(kernel='linear', random_state=42)
    
    # Train SVM classifier
    svm_classifier.fit(X_train, y_train)
    
    return svm_classifier, X_test, y_test

def evaluate_model(svm_classifier, X_test, y_test):
    # Predict labels for test set
    y_pred = svm_classifier.predict(X_test)
    
    # Evaluate model performance
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

def main():
    # File paths
    combined_features_file = "SVM/train/combined_features.npz"
    labels_file = "SVM/train/labels.csv"
    
    # Load data
    combined_features, labels = load_data(combined_features_file, labels_file)
    
    # Train SVM model
    svm_classifier, X_test, y_test = train_svm(combined_features, labels)
    
    # Evaluate model
    evaluate_model(svm_classifier, X_test, y_test)
    
    # Save trained model
    model_file = "SVM/train/svm_model.pkl"
    joblib.dump(svm_classifier, model_file)
    print("Trained SVM model saved to", model_file)

if __name__ == "__main__":
    main()
