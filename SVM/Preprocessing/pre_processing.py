import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation and special characters
    tokens = [word for word in tokens if word.isalnum()]
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def preprocess_csv(input_csv, output_csv):
    # Read CSV file
    df = pd.read_csv(input_csv)

    # Remove songs with "pop" as a genre
    df= df[df['genre'] != 'pop']
    
    # Preprocess lyrics text
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    # Save the preprocessed data with relevant columns
    df.to_csv(output_csv, index=False)
    
if __name__ == "__main__":
    input_csv = "SVM/Preprocessing/complete_data.csv"
    output_csv = "SVM/Preprocessing/preprocessed_data.csv"
    
    preprocess_csv(input_csv, output_csv)
    print(f"Preprocessed data saved to {output_csv}")
