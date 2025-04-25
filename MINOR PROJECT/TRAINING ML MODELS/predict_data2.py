import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

# Define base path
base_path = r'C:\Users\Dell\Desktop\MINOR PROJECT\CSV Files'

# Load the TF-IDF vectorizer and best model
vectorizer_path = os.path.join(base_path, 'tfidf_vectorizer.pkl')
model_path = os.path.join(base_path, 'best_model.pkl')

# Check if files exist
if not os.path.exists(vectorizer_path):
    print(f"Error: Vectorizer file '{vectorizer_path}' not found. Please train and save the vectorizer first.")
    exit(1)
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found. Please train and save the model first.")
    exit(1)

# Load the vectorizer and model
try:
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading vectorizer or model: {e}")
    exit(1)

# Define label mapping
label_mapping = {
    0: "happy",
    1: "sad",
    2: "offensive/hateful"
}

# Improved text cleaning function (same as training)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\\x[a-fA-F0-9]{2,}', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# Function to predict sentiment for new text
def predict_sentiment(text):
    # Handle empty or invalid input
    if not isinstance(text, str) or text.strip() == "":
        return "Error: Input text must be a non-empty string.", None
    
    # Clean the input text
    cleaned_text = clean_text(text)
    
    # Vectorize the input text
    text_tfidf = vectorizer.transform([cleaned_text])
    
    # Predict the label
    prediction = model.predict(text_tfidf)[0]
    
    # Get prediction probabilities if the model supports it
    try:
        probabilities = model.predict_proba(text_tfidf)[0]
        prob_dict = {label_mapping[i]: f"{prob*100:.2f}%" for i, prob in enumerate(probabilities)}
    except AttributeError:
        prob_dict = "Probabilities not available for this model."
    
    # Map the prediction to sentiment category
    sentiment = label_mapping.get(prediction, "unknown")
    return sentiment, prob_dict

# Example usage: Predict sentiment for new texts
new_texts = [
    "I feel awful today",
    "He is looking happy today",
    "Madarchod kya hai yeh",
    ""
]

# Predict and print results
print("\nSentiment Predictions:")
for text in new_texts:
    sentiment, probabilities = predict_sentiment(text)
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {sentiment}")
    print(f"Probabilities: {probabilities}\n")