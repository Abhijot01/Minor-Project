import scipy.sparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Define paths
output_dir = r'C:\Users\Dell\Desktop\MINOR PROJECT\CSV Files'

# Load the splits
X_train_combined = scipy.sparse.load_npz(os.path.join(output_dir, 'X_train_combined.npz'))
X_test_combined = scipy.sparse.load_npz(os.path.join(output_dir, 'X_test_combined.npz'))
y_train_combined = pd.read_csv(os.path.join(output_dir, 'y_train_combined.csv'))['sentiment']
y_test_combined = pd.read_csv(os.path.join(output_dir, 'y_test_combined.csv'))['sentiment']

# Train Random Forest
rf = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
rf.fit(X_train_combined, y_train_combined)

# Evaluate the model
y_pred = rf.predict(X_test_combined)
accuracy = accuracy_score(y_test_combined, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test_combined, y_pred, zero_division=1))

# Save the model
joblib.dump(rf, os.path.join(output_dir, 'random_forest_model.pkl'))
print(f"Model saved to {os.path.join(output_dir, 'random_forest_model.pkl')}")