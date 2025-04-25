import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Define paths
base_path = r'C:\Users\Dell\Desktop\MINOR PROJECT\CSV Files'
model_path = os.path.join(base_path, 'best_model.pkl')
vectorizer_path = os.path.join(base_path, 'tfidf_vectorizer.pkl')
mapping_path = os.path.join(base_path, 'hate_mapping.pkl')
output_file = os.path.join(base_path, 'predictions_new_hindi.csv')

# Check if required files exist
for file_path in [model_path, vectorizer_path, mapping_path]:
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found in {base_path}")
        exit(1)

# Load the saved model, vectorizer, and mapping
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(mapping_path, 'rb') as f:
        hate_to_class = pickle.load(f)
except Exception as e:
    print(f"Error loading files: {e}")
    exit(1)

# Define the expanded list of texts
texts = [
    "ये लिटन की गाँड में लिप्टन की चाय डालता हूँ",
    "सही एकदम इसलिये इंसानियत को मरने मत दो",
    "भोसडीके, तू क्या समझता है अपने आप को?",
    "हमें एकजुट होकर देश को आगे ले जाना है।",
    "साले, तुझे तो मैं अभी ठोक दूंगा!",
    "प्यार और शांति से ही दुनिया बदलेगी।",
    "ये लोग बस गंदगी फैलाते हैं, निकम्मे!",
    "सब मिलकर पर्यावरण को बचाएं।",
    "चल, निकल यहाँ से, कुत्ते की औलाद!",
    "आज का दिन बहुत सुंदर है, चलो खुशियां बांटें।"
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

# Map numerical predictions to labels
predicted_labels = [hate_to_class[pred] for pred in predictions]

# Create output DataFrame
output_df = pd.DataFrame({
    'text': texts,
    'predicted_label': predicted_labels
})

# Save predictions to CSV
try:
    output_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
except Exception as e:
    print(f"Error saving predictions: {e}")
    exit(1)

# Print predictions to terminal
print("\nPrediction Results:")
for text, label in zip(texts, predicted_labels):
    print(f"Text: {text[:50]}... (truncated)" if len(text) > 50 else f"Text: {text}")
    print(f"Predicted Label: {label}\n")