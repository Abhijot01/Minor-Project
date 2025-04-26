import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import os
import numpy as np
import joblib
from imblearn.over_sampling import SMOTE
import re
import matplotlib.pyplot as plt  # Added for plotting

# Define base path and input file
base_path = r'C:\Users\Dell\Desktop\MINOR PROJECT\CSV Files'
input_file = os.path.join(base_path, 'hate_cleaned_output2.csv')

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
expected_columns = ['hate', 'text', 'cleaned_text']
missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    print(f"Error: Missing columns: {missing_columns}")
    exit(1)

# Improved text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove special characters, emojis, and hex codes (e.g., xfxfxx)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\\x[a-fA-F0-9]{2,}', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# Apply improved cleaning to cleaned_text
df['cleaned_text'] = df['cleaned_text'].apply(clean_text)

# Prepare features (X) and target (y)
X = df['cleaned_text']
y = df['hate']

# Check class distribution
print("\nClass Distribution:")
print(y.value_counts(normalize=True) * 100)

# Split the data into train (80%) and test (20%) sets, stratified by hate
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Print split sizes
print(f"\nX_train size: {len(X_train)} samples")
print(f"X_test size: {len(X_test)} samples")
print(f"y_train size: {len(y_train)} samples")
print(f"y_test size: {len(y_test)} samples")

# Save splits to CSV files
X_train_df = pd.DataFrame({'cleaned_text': X_train})
X_test_df = pd.DataFrame({'cleaned_text': X_test})
y_train_df = pd.DataFrame({'hate': y_train})
y_test_df = pd.DataFrame({'hate': y_test})

# Define output paths
x_train_output = os.path.join(base_path, 'X_train_output2.csv')
x_test_output = os.path.join(base_path, 'X_test_output2.csv')
y_train_output = os.path.join(base_path, 'y_train_output2.csv')
y_test_output = os.path.join(base_path, 'y_test_output2.csv')

# Save the splits
X_train_df.to_csv(x_train_output, index=False)
X_test_df.to_csv(x_test_output, index=False)
y_train_df.to_csv(y_train_output, index=False)
y_test_df.to_csv(y_test_output, index=False)

print(f"\nSaved splits to: {base_path}")
print(f" - {x_train_output}")
print(f" - {x_test_output}")
print(f" - {y_train_output}")
print(f" - {y_test_output}")

# Handle missing values
X_train = X_train.fillna('')
X_test = X_test.fillna('')

# Vectorize text with n-grams
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_tfidf, y_train = smote.fit_resample(X_train_tfidf, y_train)

# Print new class distribution after SMOTE
print("\nClass Distribution after SMOTE (Training Set):")
print(pd.Series(y_train).value_counts(normalize=True) * 100)

# Define models with tuned hyperparameters
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, C=1.0),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, max_depth=20),
    'Naive Bayes': MultinomialNB(alpha=0.5),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42, max_depth=10),
    'SVM': LinearSVC(class_weight='balanced', random_state=42, C=0.5)
}

# Initialize results
results = []

# Print header
print("\nHate Cleaned Output 2 Dataset")

# Train and evaluate models
for model_name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=1)
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
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['0 (happy)', '1 (sad)', '2 (offensive/hateful)'], zero_division=1))

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

# Print comparison table
print("\nModel Comparison Table:")
print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Composite Score']]
      .to_string(index=False, float_format='{:,.2f}%'.format))

# Select best model
best_model_name = results_df.iloc[0]['Model']
best_composite_score = results_df.iloc[0]['Composite Score']

# Print best model
print(f"\nBest Model: {best_model_name}")
print(f"Composite Score: {best_composite_score:.2f}%")
print("\nNote: Best model based on Composite Score (average of normalized metrics). Review all metrics.")

# Save the vectorizer and best model with hindi_hate prefix
vectorizer_path = os.path.join(base_path, 'hate_cleaned2_tfidf_vectorizer.pkl')
model_path = os.path.join(base_path, 'hate_cleaned2_best_model.pkl')

joblib.dump(vectorizer, vectorizer_path)
joblib.dump(models[best_model_name], model_path)

print(f"\nSaved vectorizer to: {vectorizer_path}")
print(f"Saved best model ({best_model_name}) to: {model_path}")

# --- Plotting Section (Line Graph) ---

# Prepare data for plotting
models_list = results_df['Model'].tolist()
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['blue', 'green', 'red', 'purple']  # Colors for each metric
markers = ['o', 's', '^', 'd']  # Markers for each metric

try:
    plt.figure(figsize=(10, 6))
    for idx, metric in enumerate(metrics_to_plot):
        plt.plot(models_list, results_df[metric], marker=markers[idx], color=colors[idx], label=metric, linestyle='-')
    plt.title('Performance Comparison of Machine Learning Models (Hindi Hate Dataset)')
    plt.xlabel('Models')
    plt.ylabel('Score (%)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    line_graph_path = os.path.join(base_path, 'hate_line_graph_performance.png')
    plt.savefig(line_graph_path)
    print(f"Line graph saved to: {line_graph_path}")
    plt.show()
    plt.close()
except Exception as e:
    print(f"Error generating Line Graph: {str(e)}")