import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pickle
import uuid

# Define paths
base_path = r'C:\Users\Dell\Desktop\MINOR PROJECT\CSV Files'
input_file = os.path.join(base_path, 'cleaned_data4.csv')
x_train_path = os.path.join(base_path, 'X_train1.csv')
x_test_path = os.path.join(base_path, 'X_test1.csv')
y_train_path = os.path.join(base_path, 'y_train1.csv')
y_test_path = os.path.join(base_path, 'y_test1.csv')
model_save_path = os.path.join(base_path, 'best_model_sentiment3.pkl')
vectorizer_save_path = os.path.join(base_path, 'vectorizer3.pkl')

# Check if input file exists
if not os.path.exists(input_file):
    print(f"Error: Input file '{input_file}' not found in {base_path}")
    exit(1)

# Load dataset
try:
    df = pd.read_csv(input_file)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# Drop unnecessary columns
df = df[['cleaned_text', 'class']]

# Handle missing values
df['cleaned_text'] = df['cleaned_text'].fillna('')

# Validate class values
if not df['class'].isin([0, 1, 2]).all():
    print("Error: Invalid class values in 'class' column")
    exit(1)

# Prepare features and target
X = df['cleaned_text']
y = df['class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save split files
try:
    pd.DataFrame({'cleaned_text': X_train}).to_csv(x_train_path, index=False)
    pd.DataFrame({'cleaned_text': X_test}).to_csv(x_test_path, index=False)
    pd.DataFrame({'class': y_train}).to_csv(y_train_path, index=False)
    pd.DataFrame({'class': y_test}).to_csv(y_test_path, index=False)
    print("Split files saved successfully.")
except Exception as e:
    print(f"Error saving split files: {e}")
    exit(1)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save vectorizer
try:
    with open(vectorizer_save_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved as {vectorizer_save_path}")
except Exception as e:
    print(f"Error saving vectorizer: {e}")
    exit(1)

# Define models
models = {
    'Logistic Regression': LogisticRegression(multi_class='multinomial', class_weight='balanced', max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'Naive Bayes': MultinomialNB(),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
    'SVM': LinearSVC(class_weight='balanced', random_state=42)
}

# Initialize results
results = []

# Print header
print("Training Models on cleaned_data4.csv")

# Train and evaluate models
for model_name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
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

# Save best model
try:
    with open(model_save_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Best model ({best_model_name}) saved as {model_save_path}")
except Exception as e:
    print(f"Error saving model: {e}")
    exit(1)

# Print comparison table
print("\nModel Comparison Table:")
print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Composite Score']]
      .to_string(index=False, float_format='{:,.2f}%'.format))

# Print best model
print(f"\nBest Model: {best_model_name}")
print(f"Composite Score: {best_composite_score:.2f}%")
print("\nNote: Best model based on Composite Score (average of normalized metrics). Review all metrics.")