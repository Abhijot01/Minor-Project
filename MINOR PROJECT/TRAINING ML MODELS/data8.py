import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import os
import pickle
import matplotlib.pyplot as plt  # Added for plotting

# Define paths
base_path = r'C:\Users\Dell\Desktop\MINOR PROJECT\CSV Files'
input_file = os.path.join(base_path, 'cleaned_file (sentiments).csv')
xtrain_file = os.path.join(base_path, 'xtrain_senti2.csv')
ytrain_file = os.path.join(base_path, 'ytrain_senti2.csv')
xtest_file = os.path.join(base_path, 'xtest_senti2.csv')
ytest_file = os.path.join(base_path, 'ytest_senti2.csv')
model_path = os.path.join(base_path, 'best_model_sentiment_2.pkl')
vectorizer_path = os.path.join(base_path, 'tfidf_vectorizer_sentiment_2.pkl')
label_encoder_path = os.path.join(base_path, 'label_encoder_sentiment_2.pkl')

# Check if input file exists
if not os.path.exists(input_file):
    print(f"Error: File '{input_file}' not found in {base_path}")
    exit(1)

# Load dataset
try:
    df = pd.read_csv(input_file)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# Verify required columns
required_columns = ['text', 'sentiment']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Error: Missing columns: {missing_columns}")
    exit(1)

# Handle missing values
df['text'] = df['text'].fillna('').astype(str)
df = df[df['text'].str.strip() != '']
df = df[df['sentiment'].notnull()]

# Prepare features and labels
X = df['text']
y = df['sentiment']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split and save files
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
try:
    pd.DataFrame({'text': X_train}).to_csv(xtrain_file, index=False)
    pd.DataFrame({'sentiment': y_train}).to_csv(ytrain_file, index=False)
    pd.DataFrame({'text': X_test}).to_csv(xtest_file, index=False)
    pd.DataFrame({'sentiment': y_test}).to_csv(ytest_file, index=False)
    print("Split files saved successfully.")
except Exception as e:
    print(f"Error saving split files: {e}")
    exit(1)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

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
print("Sentiment Analysis Dataset")

# Train and evaluate models
for model_name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
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

# Print comparison table
print("\nModel Comparison Table:")
print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Composite Score']]
      .to_string(index=False, float_format='{:,.2f}%'.format))

# Print best model
print(f"\nBest Model: {best_model_name}")
print(f"Composite Score: {best_composite_score:.2f}%")
print("\nNote: Best model based on Composite Score (average of normalized metrics). Review all metrics.")

# Save the best model, vectorizer, and label encoder
try:
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"\nSaved artifacts to {base_path}:")
    print(f" - Best Model: {model_path}")
    print(f" - TF-IDF Vectorizer: {vectorizer_path}")
    print(f" - Label Encoder: {label_encoder_path}")
except Exception as e:
    print(f"Error saving artifacts: {e}")
    exit(1)

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
    plt.title('Performance Comparison of Machine Learning Models (Sentiment Analysis)')
    plt.xlabel('Models')
    plt.ylabel('Score (%)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    line_graph_path = os.path.join(base_path, 'sentiment_analysis_line_graph.png')
    plt.savefig(line_graph_path)
    print(f"Line graph saved to: {line_graph_path}")
    plt.show()
    plt.close()
except Exception as e:
    print(f"Error generating Line Graph: {str(e)}")