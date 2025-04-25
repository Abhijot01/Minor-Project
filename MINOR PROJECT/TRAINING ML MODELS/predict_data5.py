import pandas as pd
import os
import pickle
import numpy as np

# Define paths
base_path = r'C:\Users\Dell\Desktop\MINOR PROJECT\CSV Files'
model_path = os.path.join(base_path, 'best_model_sentiment.pkl')
vectorizer_path = os.path.join(base_path, 'tfidf_vectorizer_sentiment.pkl')
label_encoder_path = os.path.join(base_path, 'label_encoder_sentiment.pkl')

# Check if required files exist
for file_path in [model_path, vectorizer_path, label_encoder_path]:
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found in {base_path}")
        exit(1)

# Load the saved model, vectorizer, and label encoder
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
except Exception as e:
    print(f"Error loading files: {e}")
    exit(1)

# Define sample texts for prediction
texts = [
    "What a wonderful day to explore the park!",
    "I'm so frustrated with this broken phone.",
    "Just finished a relaxing yoga session.",
    "Excited for the new movie release tonight!",
    "Feeling neutral about the upcoming meeting.",
    "The beauty of the sunset left me in awe.",
    "Dealing with online hate is really tough.",
    "Overwhelmed by all these assignments."
]

# Transform texts using the loaded vectorizer
try:
    X_tfidf = vectorizer.transform(texts)
except Exception as e:
    print(f"Error transforming texts: {e}")
    exit(1)

# Make predictions
try:
    predictions = model.predict(X_tfidf)
except Exception as e:
    print(f"Error making predictions: {e}")
    exit(1)

# Map predictions to sentiment labels
try:
    predicted_labels = label_encoder.inverse_transform(predictions)
except Exception as e:
    print(f"Error mapping predictions: {e}")
    exit(1)

# Print predictions to terminal
print("\nPrediction Results:")
for text, label in zip(texts, predicted_labels):
    print(f"Text: {text[:50]}... (truncated)" if len(text) > 50 else f"Text: {text}")
    print(f"Predicted Sentiment: {label}\n")