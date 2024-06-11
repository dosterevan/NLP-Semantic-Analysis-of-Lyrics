import pandas as pd
from scipy import sparse
import joblib

def load_validation_data(validation_csv):
    return pd.read_csv(validation_csv)

def load_model_and_vectorizer(model_file, vectorizer_file):
    svm_model = joblib.load(model_file)
    tfidf_vectorizer = joblib.load(vectorizer_file)
    return svm_model, tfidf_vectorizer

def predict_and_save_results(validation_csv, model_file, vectorizer_file, results_csv):
    # Load validation data
    validation_data = load_validation_data(validation_csv)
    
    # Load trained model and vectorizer
    svm_model, tfidf_vectorizer = load_model_and_vectorizer(model_file, vectorizer_file)
    
    # Transform validation text data to TF-IDF features
    tfidf_matrix = tfidf_vectorizer.transform(validation_data['text'])
    
    # Make predictions
    genre_predictions = svm_model.predict(tfidf_matrix)
    
    # Add predictions to the validation data
    validation_data['predicted_genre'] = genre_predictions
    
    # Save the results to a new CSV file
    validation_data.to_csv(results_csv, index=False)
    print(f"Results saved to {results_csv}")

if __name__ == "__main__":
    validation_csv = "validation_data.csv"
    model_file = "svm_model.pkl"
    vectorizer_file = "../tfidf_vectorizer.pkl"
    results_csv = "validation_results.csv"
    
    predict_and_save_results(validation_csv, model_file, vectorizer_file, results_csv)
