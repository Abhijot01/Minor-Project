import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import os

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to clean text
def clean_text(text):
    # Lowercasing
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove punctuation
    tokens = [word for word in tokens if word.isalnum()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Load your data
file_path = 'tweets_dataset_cleaned.csv'
data = pd.read_csv(file_path)

# Apply cleaning to the 'tweet' column
data['cleaned_text'] = data['tweet'].apply(clean_text)

# Save the cleaned data to a new CSV file
output_file_path = os.path.join(os.path.dirname(file_path), 'cleaned_data.csv')
data.to_csv(output_file_path, index=False)

print(f"Cleaned data saved to {output_file_path}")