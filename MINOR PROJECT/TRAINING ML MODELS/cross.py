import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# Step 1: Define the base path and load the datasets
base_path = r'C:\Users\Dell\Desktop\MINOR PROJECT\CSV Files'

# List of the 5 CSV files (adjust file names if they are different)
dataset_files = [
    'cleaned_sentiments_preprocessed.csv',
    'cleaned_data4.csv',
    'hate_cleaned_output2.csv',
    'hindi_cleaned_output3.csv',
    'cleaned_output1.csv'
]

# Load the datasets into a dictionary
datasets = {}
for i, file_name in enumerate(dataset_files, 1):
    file_path = os.path.join(base_path, file_name)
    try:
        df = pd.read_csv(file_path)
        # Assuming the target column is named 'target'; adjust if different
        X = df.drop("target", axis=1)  # Features
        y = df["target"]               # Target
        datasets[f"Dataset{i}"] = (X, y)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found. Please check the file path.")
        exit(1)
    except KeyError:
        print(f"Error: Column 'target' not found in {file_name}. Please check the column name.")
        exit(1)

# Step 2: Define the models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "SVM": SVC(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB()
}

# Step 3: Cross-validation function
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1-Score": f1_score(y_test, y_pred, average='weighted')
    }
    return metrics

# Step 4: Cross-dataset evaluation
results = []
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

for train_name, (X_train, y_train) in datasets.items():
    print(f"\nTraining on {train_name}:")
    
    for model_name, model in models.items():
        # Perform 5-fold cross-validation on the training dataset
        cv_scores = cross_validate(model, X_train, y_train, cv=5, scoring=scoring, return_train_score=False)
        
        print(f"\nModel: {model_name}")
        print(f"Cross-Validation (on {train_name}):")
        print(f"  CV Accuracy: {np.mean(cv_scores['test_accuracy']):.3f}")
        print(f"  CV Precision: {np.mean(cv_scores['test_precision_weighted']):.3f}")
        print(f"  CV Recall: {np.mean(cv_scores['test_recall_weighted']):.3f}")
        print(f"  CV F1-Score: {np.mean(cv_scores['test_f1_weighted']):.3f}")
        
        # Evaluate on other datasets
        for test_name, (X_test, y_test) in datasets.items():
            if train_name != test_name:
                metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
                results.append({
                    "Model": model_name,
                    "Trained on": train_name,
                    "Tested on": test_name,
                    "Accuracy": metrics["Accuracy"],
                    "Precision": metrics["Precision"],
                    "Recall": metrics["Recall"],
                    "F1-Score": metrics["F1-Score"]
                })
                print(f"  Tested on {test_name}:")
                print(f"    Accuracy: {metrics['Accuracy']:.3f}")
                print(f"    Precision: {metrics['Precision']:.3f}")
                print(f"    Recall: {metrics['Recall']:.3f}")
                print(f"    F1-Score: {metrics['F1-Score']:.3f}")

# Step 5: Summarize results in a DataFrame
results_df = pd.DataFrame(results)
print("\nSummary of Cross-Dataset Evaluation:")
print(results_df)

# Step 6: Average performance per model across all datasets
avg_performance = results_df.groupby("Model").mean(numeric_only=True)
print("\nAverage Performance per Model Across All Datasets:")
print(avg_performance)



# NOT WORKING    