import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import re
import nltk
from collections import Counter

# Download stopwords and WordNet lemmatizer data
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
data = pd.read_csv('Logistic_Regression/complete_data.csv')

# Remove songs with "pop" as a genre
data = data[data['genre'] != 'pop']
data = data[data['genre'] != 'rock']

# Preprocess lyrics
def preprocess_lyrics(lyrics):
    lyrics = re.sub(r'\n', ' ', lyrics)
    lyrics = re.sub(r'[^a-zA-Z\s]', '', lyrics)
    lyrics = lyrics.lower()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = lyrics.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

data['text'] = data['text'].apply(preprocess_lyrics)

# Define features and labels
X = data['text']
y = data['seeds']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check for and handle NaN values in labels
if y_train.isna().any():
    print("Found NaN values in labels, removing them...")
    nan_indices = y_train[y_train.isna()].index
    X_train = X_train.drop(nan_indices, axis=0)
    y_train = y_train.dropna()

# Determine the smallest class size for setting k_neighbors in SMOTE
min_class_size = min(Counter(y_train).values())
if min_class_size > 1:
    k_neighbors = min(5, min_class_size - 1)  # Ensure k_neighbors is valid
else:
    k_neighbors = 1

# Define a pipeline combining SMOTE, TF-IDF vectorization, and Logistic Regression
pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 1))),  # Reduced max_features and ngram_range
    ('smote', SMOTE(random_state=42, k_neighbors=k_neighbors) if min_class_size > k_neighbors else 'passthrough'),
    ('clf', LogisticRegression(max_iter=1000))
])

# Define the parameter grid for GridSearchCV with reduced complexity
param_grid = {
    'clf__C': [1, 10],  # Reduced parameter options
}

# Initialize GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)

# Train the model
grid_search.fit(X_train, y_train)

# Get the best estimator
best_model = grid_search.best_estimator_

# Make predictions on the testing data
y_pred = best_model.predict(X_test)

# Ensure consistent data types
y_test = y_test.astype(str)
y_pred = y_pred.astype(str)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=1)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# Save the model and vectorizer
joblib.dump(best_model, 'Logistic_Regression/Emotion/best_model_emotion.pkl')

def load_resources(model_file):
    model = joblib.load(model_file)
    return model

def predict_genre(lyrics, model):
    lyrics = preprocess_lyrics(lyrics)
    prediction = model.predict([lyrics])
    return prediction[0]

def main():
    model_file = 'Logistic_Regression/Emotion/best_model_emotion.pkl'
    
    # Load the model and vectorizer
    model = load_resources(model_file)

if __name__ == '__main__':
    main()
