from imblearn.pipeline import Pipeline as ImbPipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import re
from collections import Counter

def preprocess_lyrics(lyrics):
    lyrics = re.sub(r'\n', ' ', lyrics)
    lyrics = re.sub(r'[^a-zA-Z\s]', '', lyrics)
    lyrics = lyrics.lower()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = lyrics.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)


def load_resources(model_file):
    model = joblib.load(model_file)
    return model

def predict_genre(lyrics, model):
    lyrics = preprocess_lyrics(lyrics)
    prediction = model.predict([lyrics])
    return prediction[0]

def main():
    model_file = 'Logistic_Regression/best_model.pkl'
    
    # Song With Lyrics
    WeWillRockYou =  'Logistic_Regression/songs/WeWillRockYou.txt' 
    RapGod = "Logistic_Regression/songs/RapGod.txt"
    SmellsLikeTeenSpirit = "Logistic_Regression/songs/SmellsLikeTeenSpirit.txt"

    #  # Read lyrics from a text file
    with open(WeWillRockYou, 'r') as file:
        lyrics = file.read()

    print(lyrics)
    # Load the model and vectorizer
    model = load_resources(model_file)
    
    # Predict the genre
    genre_prediction = predict_genre(lyrics, model)
    
    print(f'The predicted genre is: {genre_prediction}')

if __name__ == '__main__':
    main()