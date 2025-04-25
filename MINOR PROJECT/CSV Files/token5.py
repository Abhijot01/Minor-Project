import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import emoji
import os

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define contractions dictionary
contractions = {
    "won't": "will not",
    "can't": "cannot",
    "i'm": "i am",
    "you're": "you are",
    "it's": "it is",
    "don't": "do not",
    "didn't": "did not",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "we're": "we are",
    "they're": "they are",
    "let's": "let us",
    "that's": "that is",
    "what's": "what is",
    "there's": "there is",
    "hasn't": "has not",
    "haven't": "have not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "isn't": "is not",
    "aren't": "are not"
}

# Initialize stopword list and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define preprocessing function
def preprocess_text(text):
    # Handle None or non-string input
    if not isinstance(text, str):
        return ""
    
    # Expand contractions
    for contraction, expanded in contractions.items():
        text = text.lower().replace(contraction.lower(), expanded)
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace emojis with their descriptions
    text = emoji.demojize(text, delimiters=(" ", " "))
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove special characters, punctuation, and extra spaces, keep alphanumeric and hashtags
    text = re.sub(r'[^a-z0-9\s#]', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Process hashtags and remove stopwords, lemmatize
    cleaned_tokens = []
    for token in tokens:
        # Handle hashtags
        if token.startswith('#'):
            # Remove # and treat as a word
            token = token[1:].lower()
            if token:  # Ensure not empty
                # Split camelCase hashtags (e.g., #TeenCrush → teen crush)
                split_tokens = re.findall(r'[a-z]+', token)
                for split_token in split_tokens:
                    if split_token not in stop_words:
                        cleaned_tokens.append(lemmatizer.lemmatize(split_token))
        else:
            # Remove stopwords and lemmatize
            if token not in stop_words and token.isalnum():
                cleaned_tokens.append(lemmatizer.lemmatize(token))
    
    # Join tokens back into a string
    return ' '.join(cleaned_tokens)

# Define input and output paths
base_path = r'C:\Users\Dell\Desktop\MINOR PROJECT\CSV Files'
input_file = os.path.join(base_path, 'cleaned_file (sentiments).csv')
output_file = os.path.join(base_path, 'cleaned_sentiments_preprocessed.csv')

# Check if input file exists
if not os.path.exists(input_file):
    print(f"Error: File '{input_file}' not found")
    exit(1)

# Load the dataset
try:
    df = pd.read_csv(input_file)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# Verify expected columns
expected_columns = ['text', 'sentiment']
missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    print(f"Error: Missing columns: {missing_columns}")
    exit(1)

# Apply preprocessing
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Remove rows where cleaned_text is empty
df = df[df['cleaned_text'].str.strip() != '']

# Save the preprocessed dataset
try:
    df.to_csv(output_file, index=False)
    print(f"Preprocessed dataset saved to: {output_file}")
except Exception as e:
    print(f"Error saving file: {e}")
    exit(1)

# Print sample of preprocessed data
print("\nSample of Preprocessed Data:")
print(df[['text', 'cleaned_text', 'sentiment']].head(10).to_string(index=False))