import pandas as pd
import pickle
import re
import os

# Define paths
base_path = r'C:\Users\Dell\Desktop\MINOR PROJECT\CSV Files'
model_path = os.path.join(base_path, 'best_model.pkl')
vectorizer_path = os.path.join(base_path, 'tfidf_vectorizer.pkl')
mapping_path = os.path.join(base_path, 'sentiment_mapping.pkl')

# Check if required files exist
for path in [model_path, vectorizer_path, mapping_path]:
    if not os.path.exists(path):
        print(f"Error: File '{path}' not found in {base_path}")
        exit(1)

# Load the vectorizer, model, and mapping
try:
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(mapping_path, 'rb') as f:
        sentiment_to_class = pickle.load(f)
except Exception as e:
    print(f"Error loading files: {e}")
    exit(1)

# Create reverse mapping (class to sentiment)
class_to_sentiment = {v: k for k, v in sentiment_to_class.items()}

# Basic text cleaning function (consistent with training)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove special characters and extra spaces
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# Function to predict sentiment for new texts
def predict_sentiment(texts):
    # Clean the texts
    cleaned_texts = [clean_text(text) for text in texts]
    
    # Vectorize the texts
    texts_tfidf = vectorizer.transform(cleaned_texts)
    
    # Predict numerical classes
    predictions = model.predict(texts_tfidf)
    
    # Map numerical classes back to sentiments
    predicted_sentiments = [class_to_sentiment[pred] for pred in predictions]
    
    return predicted_sentiments

# Example usage
if __name__ == "__main__":
    # Example texts for prediction
    new_texts = [
        "I had an amazing day at the beach with friends!",
        "Feeling so sad after a terrible exam result.",
        "Just attended a workshop to learn about sustainability.",
        "The concert was a complete disaster, I'm so upset.",
        "Trying out a new recipe tonight, let's see how it goes."
    ]
    
    # Make predictions
    predicted_sentiments = predict_sentiment(new_texts)
    
    # Print results
    print("\nSentiment Predictions:")
    for text, sentiment in zip(new_texts, predicted_sentiments):
        print(f"Text: {text}")
        print(f"Predicted Sentiment: {sentiment}\n")