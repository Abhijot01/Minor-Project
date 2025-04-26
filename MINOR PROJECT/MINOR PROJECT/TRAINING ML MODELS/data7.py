import pandas as pd
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# hindi_hateful.csv data

# Define paths
base_path = r'C:\Users\Dell\Desktop\MINOR PROJECT\CSV Files'
xtrain_file = os.path.join(base_path, 'xtrain_hindi.csv')
ytrain_file = os.path.join(base_path, 'ytrain_hindi.csv')
xtest_file = os.path.join(base_path, 'xtest_hindi.csv')
ytest_file = os.path.join(base_path, 'ytest_hindi.csv')
model_path = os.path.join(base_path, 'best_model.pkl')
vectorizer_path = os.path.join(base_path, 'tfidf_vectorizer.pkl')
mapping_path = os.path.join(base_path, 'hate_mapping.pkl')

# Check if input files exist
for file_path in [xtrain_file, ytrain_file, xtest_file, ytest_file]:
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found in {base_path}")
        exit(1)

# Load the datasets
try:
    X_train = pd.read_csv(xtrain_file)
    y_train = pd.read_csv(ytrain_file)
    X_test = pd.read_csv(xtest_file)
    y_test = pd.read_csv(ytest_file)
except Exception as e:
    print(f"Error loading files: {e}")
    exit(1)

# Handle missing values in text
X_train['text'] = X_train['text'].fillna('')
X_test['text'] = X_test['text'].fillna('')

# Verify text_id alignment (optional, for debugging)
if not (X_train['text_id'].equals(y_train['text_id']) and X_test['text_id'].equals(y_test['text_id'])):
    print("Warning: Mismatch in text_id between X and y files")

# Prepare features and labels
X_train_text = X_train['text']
y_train_labels = y_train['hate']
X_test_text = X_test['text']
y_test_labels = y_test['hate']

# Create and save hate-to-class mapping
hate_to_class = {0: 'Non-Hateful', 1: 'Hateful'}
with open(mapping_path, 'wb') as f:
    pickle.dump(hate_to_class, f)

# Vectorize text (no stop_words for Hindi text)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

# Save the vectorizer for prediction
with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)

# Check class distribution for debugging
print("Class Distribution:")
print(f"y_train classes: {sorted(set(y_train_labels))}")
print(f"y_train class counts:\n{y_train_labels.value_counts().sort_index()}")
print(f"y_test classes: {sorted(set(y_test_labels))}")
print(f"y_test class counts:\n{y_test_labels.value_counts().sort_index()}")
if set(y_train_labels) != set(y_test_labels):
    print("Warning: Class mismatch between y_train and y_test")

# Define models
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'Naive Bayes': MultinomialNB(),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
    'SVM': LinearSVC(class_weight='balanced', random_state=42)
}

# Initialize results
results = []

# Print header
print("\nHindi Hate Speech Training Results")

# Train and evaluate models
for model_name, model in models.items():
    model.fit(X_train_tfidf, y_train_labels)
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate metrics with zero_division to avoid warnings
    accuracy = accuracy_score(y_test_labels, y_pred) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_labels, y_pred, average='weighted', zero_division=1
    )
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

# Save the best model
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)

# Print comparison table
print("\nModel Comparison Table:")
print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Composite Score']]
      .to_string(index=False, float_format='{:,.2f}%'.format))

# Print best model
print(f"\nBest Model: {best_model_name}")
print(f"Composite Score: {best_composite_score:.2f}%")
print("\nNote: Best model based on Composite Score (average of normalized metrics). Review all metrics.")