import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy import sparse
import joblib
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import word_tokenize, pos_tag
from textblob import TextBlob
import numpy as np

nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

def extract_features(df):
    # Initialize TF-IDF vectorizer with bigrams and trigrams
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_text'])

    # Sentiment analysis using VADER
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = df['clean_text'].apply(lambda x: sid.polarity_scores(x))
    sentiment_df = pd.DataFrame(list(sentiment_scores))
    
    # TextBlob for polarity and subjectivity
    df['polarity'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    
    # Part-of-Speech tagging
    df['pos_tags'] = df['clean_text'].apply(lambda x: " ".join([pos for word, pos in pos_tag(word_tokenize(x))]))
    
    # Vectorize POS tags
    pos_vectorizer = TfidfVectorizer()
    pos_matrix = pos_vectorizer.fit_transform(df['pos_tags'])
    
    # Text length features
    df['word_count'] = df['clean_text'].apply(lambda x: len(word_tokenize(x)))
    df['sentence_count'] = df['clean_text'].apply(lambda x: len(nltk.sent_tokenize(x)))
    df['lexical_diversity'] = df['clean_text'].apply(lambda x: len(set(word_tokenize(x))) / len(word_tokenize(x)) if len(word_tokenize(x)) > 0 else 0)

    # Combine all features into a single sparse matrix
    additional_features = df[['polarity', 'subjectivity', 'word_count', 'sentence_count', 'lexical_diversity']]
    scaler = StandardScaler()
    scaled_additional_features = scaler.fit_transform(additional_features)
    additional_features_matrix = sparse.csr_matrix(scaled_additional_features)
    
    combined_features = sparse.hstack([tfidf_matrix, pos_matrix, additional_features_matrix])
    
    return combined_features, tfidf_vectorizer, pos_vectorizer, scaler

def main():
    # Read preprocessed CSV file
    preprocessed_csv = "SVM/Preprocessing/preprocessed_data.csv"
    df = pd.read_csv(preprocessed_csv)
    
    # Extract features
    combined_features, tfidf_vectorizer, pos_vectorizer, scaler = extract_features(df)
    
    # Save combined features matrix and vectorizers
    combined_features_file = "SVM/feature_extraction/combined_features.npz"
    tfidf_vectorizer_file = "SVM/feature_extraction/tfidf_vectorizer.pkl"
    pos_vectorizer_file = "SVM//feature_extraction/pos_vectorizer.pkl"
    scaler_file = "SVM/feature_extraction/scaler.pkl"
    
    sparse.save_npz(combined_features_file, combined_features)
    joblib.dump(tfidf_vectorizer, tfidf_vectorizer_file)
    joblib.dump(pos_vectorizer, pos_vectorizer_file)
    joblib.dump(scaler, scaler_file)
    
    # Save labels to a separate file for training
    labels_file = "SVM/feature_extraction/labels.csv"
    df[['artist', 'track', 'seeds', 'genre', 'clean_text']].to_csv(labels_file, index=False)
    print(f"Combined features saved to {combined_features_file}")
    print(f"TF-IDF vectorizer saved to {tfidf_vectorizer_file}")
    print(f"POS vectorizer saved to {pos_vectorizer_file}")
    print(f"Scaler saved to {scaler_file}")
    print(f"Labels saved to {labels_file}")

if __name__ == "__main__":
    main()
