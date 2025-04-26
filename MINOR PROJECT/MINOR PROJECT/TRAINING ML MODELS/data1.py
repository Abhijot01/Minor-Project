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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # Added for plotting

# Define paths
base_path = r'C:\Users\Dell\Desktop\MINOR PROJECT\CSV Files'
input_file = os.path.join(base_path, 'cleaned_output1.csv')
model_path = os.path.join(base_path, 'best_model.pkl')
vectorizer_path = os.path.join(base_path, 'tfidf_vectorizer.pkl')

# Check if input file exists
if not os.path.exists(input_file):
    print(f"Error: File '{input_file}' not found in {base_path}")
    exit(1)

# Load the dataset
try:
    df = pd.read_csv(input_file)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# Handle missing values
df['cleaned_text'] = df['cleaned_text'].fillna('')

# Map sentiment labels to numerical classes
unique_sentiments = sorted(df['sentiment'].unique())
sentiment_to_class = {sentiment: idx for idx, sentiment in enumerate(unique_sentiments)}

df['class'] = df['sentiment'].map(sentiment_to_class)

# Save the sentiment-to-class mapping for later use
mapping_path = os.path.join(base_path, 'sentiment_mapping.pkl')
with open(mapping_path, 'wb') as f:
    pickle.dump(sentiment_to_class, f)

# Prepare features and labels
X = df['cleaned_text']
y = df['class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save the vectorizer for prediction
with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)

# Check class distribution for debugging
print("Class Distribution:")
print(f"y_train classes: {sorted(set(y_train))}")
print(f"y_train class counts:\n{y_train.value_counts().sort_index()}")
print(f"y_test classes: {sorted(set(y_test))}")
print(f"y_test class counts:\n{y_test.value_counts().sort_index()}")
if set(y_train) != set(y_test):
    print("Warning: Class mismatch between y_train and y_test")

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
print("\nUpdated Data Training Results")

# Train and evaluate models
for model_name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate metrics with zero_division to avoid warnings
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=1
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

# --- Plotting Section ---

# Prepare data for plotting
models_list = results_df['Model'].tolist()
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['blue', 'green', 'red', 'purple']  # Colors for each metric
markers = ['o', 's', '^', 'd']  # Markers for each metric

# 1. Line Graph (Separate lines for each metric)
try:
    plt.figure(figsize=(10, 6))
    for idx, metric in enumerate(metrics_to_plot):
        plt.plot(models_list, results_df[metric], marker=markers[idx], color=colors[idx], label=metric, linestyle='-')
    plt.title('Performance Comparison of Machine Learning Models (Line Graph)')
    plt.xlabel('Models')
    plt.ylabel('Score (%)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    line_graph_path = os.path.join(base_path, 'line_graph_performance.png')
    plt.savefig(line_graph_path)
    print(f"Line graph saved to: {line_graph_path}")
    plt.show()
    plt.close()  # Close the figure to free memory
except Exception as e:
    print(f"Error generating Line Graph: {str(e)}")

# 2. Multiline Graph (With dual y-axes for two metrics)
try:
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Accuracy and F1-Score on the left y-axis
    ax1.plot(models_list, results_df['Accuracy'], marker='o', color='blue', label='Accuracy', linestyle='-')
    ax1.plot(models_list, results_df['F1-Score'], marker='d', color='purple', label='F1-Score', linestyle='-')
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Accuracy / F1-Score (%)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    ax1.set_ylim(min(results_df[['Accuracy', 'F1-Score']].min().min() - 5, 0), 100)  # Dynamic y-axis limit

    # Create a second y-axis for Precision and Recall
    ax2 = ax1.twinx()
    ax2.plot(models_list, results_df['Precision'], marker='s', color='green', label='Precision', linestyle='--')
    ax2.plot(models_list, results_df['Recall'], marker='^', color='red', label='Recall', linestyle='--')
    ax2.set_ylabel('Precision / Recall (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')
    ax2.set_ylim(min(results_df[['Precision', 'Recall']].min().min() - 5, 0), 100)  # Dynamic y-axis limit

    # Title and layout
    plt.title('Performance Comparison of Machine Learning Models (Multiline Graph)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    multiline_graph_path = os.path.join(base_path, 'multiline_graph_performance.png')
    plt.savefig(multiline_graph_path)
    print(f"Multiline graph saved to: {multiline_graph_path}")
    plt.show()
    plt.close()  # Close the figure to free memory
except Exception as e:
    print(f"Error generating Multiline Graph: {str(e)}")