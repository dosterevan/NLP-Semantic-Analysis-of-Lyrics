import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from scipy import sparse
import nltk
from nltk import word_tokenize, pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd
import numpy as np

def load_resources(model_file, tfidf_vectorizer_file, pos_vectorizer_file, scaler_file):
    # Load the trained model
    svm_model = joblib.load(model_file)
    
    # Load the TF-IDF vectorizer
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_file)
    
    # Load the POS vectorizer
    pos_vectorizer = joblib.load(pos_vectorizer_file)

    scaler = joblib.load(scaler_file)
    
    return svm_model, tfidf_vectorizer, pos_vectorizer, scaler

def preprocess_lyrics(lyrics):
    lyrics = re.sub(r'\n', ' ', lyrics)
    lyrics = re.sub(r'[^a-zA-Z\s]', '', lyrics)
    lyrics = lyrics.lower()
    return lyrics

def predict_seed(lyrics, model, tfidf_vectorizer, pos_vectorizer, scaler):
    # Preprocess the lyrics
    lyrics = preprocess_lyrics(lyrics)
    
    # Transform the lyrics with tfidf_vectorizer
    lyrics_tfidf = tfidf_vectorizer.transform([lyrics])
    
    # Extract POS tags from the lyrics
    pos_tags = " ".join([pos for word, pos in pos_tag(word_tokenize(lyrics))])
    
    # Transform the POS tags with pos_vectorizer
    pos_tfidf = pos_vectorizer.transform([pos_tags])
    
    # Calculate the additional features
    polarity = TextBlob(lyrics).sentiment.polarity
    subjectivity = TextBlob(lyrics).sentiment.subjectivity
    word_count = len(word_tokenize(lyrics))
    sentence_count = len(nltk.sent_tokenize(lyrics))
    lexical_diversity = len(set(word_tokenize(lyrics))) / len(word_tokenize(lyrics)) if len(word_tokenize(lyrics)) > 0 else 0
    
    # Create a DataFrame with the additional features
    additional_features_df = pd.DataFrame(data=[[polarity, subjectivity, word_count, sentence_count, lexical_diversity]], 
                                          columns=['polarity', 'subjectivity', 'word_count', 'sentence_count', 'lexical_diversity'])
    
    # Scale the additional features
    additional_features = scaler.transform(additional_features_df)
    
    # Combine the TF-IDF features and additional features
    combined_features = sparse.hstack([lyrics_tfidf, pos_tfidf, additional_features])
    
    # # Check if the number of features matches the SVM model's expectation
    # if combined_features.shape[1] != model.coef_.shape[1]:
    #     raise ValueError(f"Input data has {combined_features.shape[1]} features, but the model expects {model.coef_.shape[1]} features.")
    
    # Save the combined features to a file
    combined_features_file = "combined_features.npz"
    sparse.save_npz(combined_features_file, combined_features)
    
    # Predict the genre
    prediction = model.predict(combined_features)
    
    return prediction[0]

def main():
    model_file = "SVM/train/svm_model.pkl"
    tfidf_vectorizer_file = "SVM/train/tfidf_vectorizer.pkl"
    pos_vectorizer_file = "SVM/train/pos_vectorizer.pkl"
    scaler_file = "SVM/train/scaler.pkl"
    
    svm_model, tfidf_vectorizer, pos_vectorizer, scaler = load_resources(model_file, tfidf_vectorizer_file, pos_vectorizer_file, scaler_file)
    
    # Song With Lyrics
    WeWillRockYou =  'Logistic_Regression/songs/WeWillRockYou.txt' 
    RapGod = "Logistic_Regression/songs/RapGod.txt"
    SmellsLikeTeenSpirit = "Logistic_Regression/songs/SmellsLikeTeenSpirit.txt"
    
    # Example lyrics prompt
    #  # Read lyrics from a text file
    with open(WeWillRockYou, 'r') as file:
        lyrics = file.read()

    # Type The lyrics in manually
#     lyrics = """ 
# Yesterday a child came out to wonder
# Caught a dragonfly inside a jar 
# Fearful when the sky was full of thunder 
# And tearful at the falling of a star 

# Then the child moved ten times round the seasons
# Skated over ten clear frozen streams 
# Words like when you're older must appease him 
# And promises of someday make his dreams

# And the seasons they go round and round 
# And the painted ponies go up and down 
# We're captive on the carousel of time 
# We can't return we can only look 
# Behind from where we came 
# And go round and round and round 
# In the circle game

# Sixteen springs and sixteen summers gone now 
# Cartwheels turn to car wheels thru the town 
# And they tell him take your time it won't be long now
# Till you drag your feet to slow the circles down 

# And the seasons they go round and round 
# And the painted ponies go up and down 
# We're captive on the carousel of time 
# We can't return we can only look 
# Behind from where we came 
# And go round and round and round 
# In the circle game

# So the years spin by and now the boy is twenty 
# Though his dreams have lost some grandeur coming true
# There'll be new dreams maybe better dreams and plenty
# Before the last revolving year is through

# And the seasons they go round and round 
# And the painted ponies go up and down 
# We're captive on the carousel of time 
# We can't return we can only look 
# Behind from where we came 
# And go round and round and round 
# In the circle game
# """

    # Use the input prompt for user interaction
    print(lyrics)

    seed_prediction = predict_seed(lyrics, svm_model, tfidf_vectorizer, pos_vectorizer, scaler)
    
    print(f"The predicted emotion is: {seed_prediction}")

if __name__ == "__main__":
    main()
