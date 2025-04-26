import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
import os

# Step 1: Load the CSV data
file_path = r'C:\Users\Dell\Desktop\MINOR PROJECT\CSV Files\combined_sentiments.csv'

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File {file_path} not found. Please check the file path.")
    exit(1)

# Step 2: Verify required columns
if 'cleaned_text' not in df.columns or 'sentiment' not in df.columns:
    print("Error: Required columns 'cleaned_text' or 'sentiment' not found. Available columns:", df.columns)
    exit(1)

# Step 3: Handle NaN values in 'sentiment' column
# Check for NaN values in 'sentiment'
nan_count = df['sentiment'].isna().sum()
print(f"Number of NaN values in 'sentiment' column: {nan_count}")

# Remove rows where 'sentiment' is NaN
df_cleaned = df.dropna(subset=['sentiment']).copy()

# Debug: Confirm the number of rows after dropping NaN
print(f"Original dataset size: {len(df)}")
print(f"Dataset size after dropping NaN in 'sentiment': {len(df_cleaned)}")

# Step 4: Extract features (X) and target (y) from the cleaned dataset
X = df_cleaned['cleaned_text'].fillna('')  # Replace NaN with empty string in 'cleaned_text'
y = df_cleaned['sentiment']

# Debug: Check for NaN in y after cleaning
if y.isna().sum() > 0:
    print("Error: NaN values still present in 'sentiment' after cleaning.")
    exit(1)

# Step 5: Convert text data to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = vectorizer.fit_transform(X)

# Step 6: Split the data into training and testing sets
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# Step 7: Save the splits for later use
output_dir = r'C:\Users\Dell\Desktop\MINOR PROJECT\CSV Files'
os.makedirs(output_dir, exist_ok=True)

# Save the TF-IDF matrices (sparse format)
scipy.sparse.save_npz(os.path.join(output_dir, 'X_train_combined.npz'), X_train_combined)
scipy.sparse.save_npz(os.path.join(output_dir, 'X_test_combined.npz'), X_test_combined)

# Save the target variables as CSV files
y_train_combined.to_csv(os.path.join(output_dir, 'y_train_combined.csv'), index=False)
y_test_combined.to_csv(os.path.join(output_dir, 'y_test_combined.csv'), index=False)

# Save the vectorizer for later use
import joblib
joblib.dump(vectorizer, os.path.join(output_dir, 'tfidf_vectorizer.pkl'))

# Debug: Print shapes of the splits
print(f"X_train_combined shape: {X_train_combined.shape}")
print(f"X_test_combined shape: {X_test_combined.shape}")
print(f"y_train_combined shape: {y_train_combined.shape}")
print(f"y_test_combined shape: {y_test_combined.shape}")

# Debug: Verify the distribution of sentiments in train and test sets
print("\nSentiment distribution in y_train_combined:")
print(y_train_combined.value_counts())
print("\nSentiment distribution in y_test_combined:")
print(y_test_combined.value_counts())