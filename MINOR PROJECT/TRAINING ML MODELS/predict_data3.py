import pandas as pd
import os
import pickle
import numpy as np

# Define paths
base_path = r'C:\Users\Dell\Desktop\MINOR PROJECT\CSV Files'
model_path = os.path.join(base_path, 'best_model2.pkl')
vectorizer_path = os.path.join(base_path, 'tfidf_vectorizer2.pkl')
mlb_path = os.path.join(base_path, 'mlb_classes2.pkl')
output_file = os.path.join(base_path, 'predictions2.csv')

# Check if required files exist
for file_path in [model_path, vectorizer_path, mlb_path]:
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found in {base_path}")
        exit(1)

# Load the saved model, vectorizer, and MultiLabelBinarizer
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(mlb_path, 'rb') as f:
        mlb = pickle.load(f)
except Exception as e:
    print(f"Error loading files: {e}")
    exit(1)

# Define the list of texts for prediction
texts = [
    "तू तो बेकार है, कुछ काम का नहीं!",
    "हम सब मिलकर देश को बेहतर बनाएंगे।",
    "ये लोग सिर्फ झूठ बोलते हैं, शर्म नहीं आती?",
    "प्रकृति का सम्मान करना हमारा कर्तव्य है।",
    "साले, तुझे सबक सिखाना पड़ेगा!",
    "हर इंसान में कुछ अच्छाई होती है।",
    "ये गंदे लोग समाज को बर्बाद कर रहे हैं!",
    "शिक्षा से ही देश का भविष्य उज्जवल होगा।",
    "कुत्ते की तरह भौंकना बंद कर, बेवकूफ!",
    "आज का दिन खुशियां मनाने का है।"
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

# Map binary predictions to label names
try:
    predicted_labels = []
    for pred in predictions:
        # Get indices where prediction is 1
        label_indices = np.where(pred == 1)[0]
        if len(label_indices) == 0:
            labels = ['non-hostile']  # Default if no labels predicted
        else:
            labels = mlb.classes_[label_indices]
        predicted_labels.append(','.join(labels))
except Exception as e:
    print(f"Error mapping predictions: {e}")
    exit(1)

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