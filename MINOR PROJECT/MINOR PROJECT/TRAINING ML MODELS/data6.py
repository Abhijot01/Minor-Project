import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Define base directory for input and output
base_dir = r"C:\Users\Dell\Desktop\MINOR PROJECT\CSV Files"

# Load cleaned dataset
file_path = os.path.join(base_dir, "cleaned_PMLN_predicted_tweets.csv")
try:
    data = pd.read_csv(file_path)
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    exit(1)

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "RandomForest": RandomForestClassifier(random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "NaiveBayes": MultinomialNB()
}

# Function to save split files
def save_splits(X_train, X_test, y_train, y_test, language, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, f"X_train_{language}.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, f"X_test_{language}.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, f"y_train_{language}.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, f"y_test_{language}.csv"), index=False)
    print(f"Saved splits for {language} in {output_dir}")

# Function to train and evaluate models
def train_and_evaluate(X_train, X_test, y_train, y_test, language):
    # Initialize TF-IDF vectorizer with reduced features
    vectorizer = TfidfVectorizer(max_features=2000)
    try:
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
    except Exception as e:
        print(f"Error in TF-IDF vectorization for {language}: {str(e)}")
        return []

    # Store results
    results = []

    for name, model in models.items():
        try:
            # Train model
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)

            # Calculate metrics and convert to percentages
            accuracy = accuracy_score(y_test, y_pred) * 100
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=1) * 100
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=1) * 100
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=1) * 100

            # Store results
            results.append({
                "Model": name,
                "Language": language,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1
            })
            print(f"{language} - {name}: Accuracy={accuracy:.2f}%, Precision={precision:.2f}%, Recall={recall:.2f}%, F1={f1:.2f}%")

        except Exception as e:
            print(f"Error training {name} for {language}: {str(e)}")

    return results

# Process each language
languages = ["in", "ur", "en"]
all_results = []

for lang in languages:
    print(f"\nProcessing language: {lang}")
    # Filter data by language
    lang_data = data[data["language"] == lang][["cleaned_tweet", "sentiment"]]
    
    if lang_data.empty:
        print(f"No data found for language {lang}")
        continue

    # Ensure cleaned_tweet is string and handle NaN
    lang_data["cleaned_tweet"] = lang_data["cleaned_tweet"].fillna("").astype(str)
    
    # Split data
    X = lang_data["cleaned_tweet"]
    y = lang_data["sentiment"]
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except Exception as e:
        print(f"Error splitting data for {lang}: {str(e)}")
        continue

    # Save splits in the CSV Files folder
    save_splits(X_train, X_test, y_train, y_test, lang, base_dir)

    # Train and evaluate models
    results = train_and_evaluate(X_train, X_test, y_train, y_test, lang)
    all_results.extend(results)

# Save results to a CSV file in the CSV Files folder
if all_results:
    results_df = pd.DataFrame(all_results)
    results_file = os.path.join(base_dir, "model_performance.csv")
    # Format the metrics as percentages in the CSV
    results_df["Accuracy"] = results_df["Accuracy"].map("{:.2f}%".format)
    results_df["Precision"] = results_df["Precision"].map("{:.2f}%".format)
    results_df["Recall"] = results_df["Recall"].map("{:.2f}%".format)
    results_df["F1-Score"] = results_df["F1-Score"].map("{:.2f}%".format)
    results_df.to_csv(results_file, index=False)
    print(f"\nSaved model performance to {results_file}")

    # Generate line chart for average metrics by language
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    avg_metrics = results_df.groupby("Language")[metrics].mean().reset_index()

    plt.figure(figsize=(10, 6))
    for metric in metrics:
        plt.plot(avg_metrics["Language"], avg_metrics[metric], marker='o', label=metric)
    
    plt.title("Average Model Performance by Language")
    plt.xlabel("Language")
    plt.ylabel("Score (%)")
    plt.legend()
    plt.grid(True)
    plt.xticks(avg_metrics["Language"], ["Indian English", "Urdu", "English"])
    plt.tight_layout()
    
    # Save plot to CSV Files folder
    plot_file = os.path.join(base_dir, "performance_plot.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Saved performance plot to {plot_file}")
else:
    print("\nNo results to save due to errors or empty data.")