import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import numpy as np
import pickle

# Define base path and input file
base_path = r'C:\Users\Dell\Desktop\MINOR PROJECT\CSV Files'
input_file = os.path.join(base_path, 'hindi_cleaned_output3.csv')

# Define paths for saved artifacts
model_path = os.path.join(base_path, 'best_model2.pkl')
vectorizer_path = os.path.join(base_path, 'tfidf_vectorizer2.pkl')
mlb_path = os.path.join(base_path, 'mlb_classes2.pkl')

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

# Print column names for debugging
print("Column names in DataFrame:", df.columns.tolist())

# Verify expected columns
expected_columns = ['Unique ID', 'Post', 'cleaned_text', 'Labels Set']
missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    print(f"Error: Missing columns: {missing_columns}")
    exit(1)

# Prepare features (X) and target (y)
X = df['cleaned_text']
y = df['Labels Set'].apply(lambda x: x.split(','))  # Split comma-separated labels

# Split the data into train (80%) and test (20%) sets, stratified by Labels Set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df['Labels Set']
)

# Print split sizes
print(f"X_train size: {len(X_train)} samples")
print(f"X_test size: {len(X_test)} samples")
print(f"y_train size: {len(y_train)} samples")
print(f"y_test size: {len(y_test)} samples")

# Save splits to CSV files
X_train_df = pd.DataFrame({'cleaned_text': X_train})
X_test_df = pd.DataFrame({'cleaned_text': X_test})
y_train_df = pd.DataFrame({'Labels Set': [','.join(labels) for labels in y_train]})
y_test_df = pd.DataFrame({'Labels Set': [','.join(labels) for labels in y_test]})

# Define output paths
x_train_output = os.path.join(base_path, 'X_train_output3.csv')
x_test_output = os.path.join(base_path, 'X_test_output3.csv')
y_train_output = os.path.join(base_path, 'y_train_output3.csv')
y_test_output = os.path.join(base_path, 'y_test_output3.csv')

# Save the splits
X_train_df.to_csv(x_train_output, index=False)
X_test_df.to_csv(x_test_output, index=False)
y_train_df.to_csv(y_train_output, index=False)
y_test_df.to_csv(y_test_output, index=False)

print(f"Saved splits to: {base_path}")
print(f" - {x_train_output}")
print(f" - {x_test_output}")
print(f" - {y_train_output}")
print(f" - {y_test_output}")

# Handle missing values
X_train = X_train.fillna('')
X_test = X_test.fillna('')

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Transform labels into multi-label binary format
mlb = MultiLabelBinarizer()
y_train_bin = mlb.fit_transform(y_train)
y_test_bin = mlb.transform(y_test)

# Define models
models = {
    'Logistic Regression': OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight='balanced')),
    'Random Forest': OneVsRestClassifier(RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
    'Naive Bayes': OneVsRestClassifier(MultinomialNB()),
    'Decision Tree': OneVsRestClassifier(DecisionTreeClassifier(class_weight='balanced', random_state=42)),
    'SVM': OneVsRestClassifier(LinearSVC(class_weight='balanced', random_state=42))
}

# Initialize results
results = []

# Print header
print("\nHindi Cleaned Output 3 Dataset")

# Train and evaluate models
for model_name, model in models.items():
    model.fit(X_train_tfidf, y_train_bin)
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_bin, y_pred) * 100  # Subset accuracy
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_bin, y_pred, average='weighted')
    precision *= 100
    recall *= 100
    f1 *= 100
    
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    print(f"\n{model_name} Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1-Score: {f1:.2f}%")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Print raw metrics
print("\nRaw Metrics:")
print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))

# Adjusted normalization
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
def adjusted_normalize(x):
    if x.max() == x.min():
        return np.ones_like(x)
    range_val = x.max() - x.min()
    return (x - x.min() + 0.01 * range_val) / (range_val + 0.01 * range_val)

normalized_metrics = results_df[metrics].apply(adjusted_normalize)
results_df['Composite Score'] = normalized_metrics.mean(axis=1) * 100

# Sort by Composite Score
results_df = results_df.sort_values(by='Composite Score', ascending=False)

# Select best model
best_model_name = results_df.iloc[0]['Model']
best_composite_score = results_df.iloc[0]['Composite Score']
best_model = models[best_model_name]

# Print comparison table
print("\nModel Comparison Table:")
print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Composite Score']]
      .to_string(index=False, float_format='{:,.2f}%'.format))

# Print best model
print(f"\nBest Model: {best_model_name}")
print(f"Composite Score: {best_composite_score:.2f}%")
print("\nNote: Best model based on Composite Score (average of normalized metrics). Review all metrics.")

# Save the best model, vectorizer, and MLB classes
try:
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(mlb_path, 'wb') as f:
        pickle.dump(mlb, f)
    print(f"\nSaved artifacts to {base_path}:")
    print(f" - Best Model: {model_path}")
    print(f" - TF-IDF Vectorizer: {vectorizer_path}")
    print(f" - MultiLabelBinarizer Classes: {mlb_path}")
except Exception as e:
    print(f"Error saving artifacts: {e}")
    exit(1)