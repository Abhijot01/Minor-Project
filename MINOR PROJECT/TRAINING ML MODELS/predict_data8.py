import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

# Define paths
base_path = r'C:\Users\Dell\Desktop\MINOR PROJECT\CSV Files'
xtest_file = os.path.join(base_path, 'xtest_senti2.csv')
ytest_file = os.path.join(base_path, 'ytest_senti2.csv')
model_path = os.path.join(base_path, 'best_model_sentiment_2.pkl')
vectorizer_path = os.path.join(base_path, 'tfidf_vectorizer_sentiment_2.pkl')
label_encoder_path = os.path.join(base_path, 'label_encoder_sentiment_2.pkl')

# Check if required files exist
for file_path in [xtest_file, ytest_file, model_path, vectorizer_path, label_encoder_path]:
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found in {base_path}")
        exit(1)

# Load the test data
try:
    X_test = pd.read_csv(xtest_file)
    y_test = pd.read_csv(ytest_file)
except Exception as e:
    print(f"Error loading test files: {e}")
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
    print(f"Error loading artifacts: {e}")
    exit(1)

# Verify required columns
if 'text' not in X_test.columns or 'sentiment' not in y_test.columns:
    print("Error: Required columns missing in test data")
    exit(1)

# Prepare test data
X_test_text = X_test['text'].fillna('').astype(str)
X_test_tfidf = vectorizer.transform(X_test_text)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Decode actual and predicted labels
y_test_labels = label_encoder.inverse_transform(y_test['sentiment'])
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Display results in terminal
print("\nSentiment Prediction Results:")
print("-" * 50)
for i in range(min(10, len(X_test_text))):  # Display first 10 samples for brevity
    print(f"\nText: {X_test_text.iloc[i]}")
    print(f"Actual Sentiment: {y_test_labels[i]}")
    print(f"Predicted Sentiment: {y_pred_labels[i]}")
print("\nNote: Displaying only the first 10 samples for brevity.")

# --- Plotting Section (Scatter Plot for Actual vs Predicted) ---

# Use the encoded numerical values for plotting
y_test_encoded = y_test['sentiment'].values  # Already encoded
y_pred_encoded = y_pred  # Predictions are already in encoded form

# Limit to first 50 samples for readability
num_samples = min(50, len(y_test_encoded))
x_axis = np.arange(num_samples)

# Get unique encoded values and their corresponding labels for the y-axis
unique_values = np.unique(np.concatenate([y_test_encoded[:num_samples], y_pred_encoded[:num_samples]]))
unique_labels = label_encoder.inverse_transform(unique_values)

try:
    plt.figure(figsize=(12, 6))
    plt.scatter(x_axis, y_test_encoded[:num_samples], color='blue', label='Actual Sentiment', marker='o', alpha=0.6)
    plt.scatter(x_axis, y_pred_encoded[:num_samples], color='red', label='Predicted Sentiment', marker='x', alpha=0.6)
    plt.title('Actual vs Predicted Sentiments (First 50 Samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Sentiment')
    plt.yticks(ticks=unique_values, labels=unique_labels)  # Replace numerical ticks with sentiment labels
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    graph_path = os.path.join(base_path, 'sentiment_prediction_scatter.png')
    plt.savefig(graph_path)
    print(f"\nScatter plot comparing actual vs predicted sentiments saved to: {graph_path}")
    plt.show()
    plt.close()
except Exception as e:
    print(f"Error generating Scatter Plot: {str(e)}")