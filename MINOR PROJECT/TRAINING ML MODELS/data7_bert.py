import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Define base directory
base_dir = r"C:\Users\Dell\Desktop\MINOR PROJECT\CSV Files"

# Load dataset
file_path = os.path.join(base_dir, "hindi_hateful.csv")
try:
    data = pd.read_csv(file_path)
    print(f"Loaded dataset with {len(data)} rows")
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    exit(1)

# Custom Dataset class for BERT with index fix
class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        # Convert to lists with sequential indices
        self.texts = texts.tolist()  # Ensure texts is a list
        self.labels = labels.tolist()  # Ensure labels is a list
        self.tokenizer = tokenizer
        self.max_length = max_length
        assert len(self.texts) == len(self.labels), "Mismatch between texts and labels"

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Function to train and evaluate BERT
def train_and_evaluate_bert(X_train, X_test, y_train, y_test):
    # Load pre-trained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create datasets and dataloaders
    train_dataset = HateSpeechDataset(X_train, y_train, tokenizer)
    test_dataset = HateSpeechDataset(X_test, y_test, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Fine-tune BERT
    optimizer = AdamW(model.parameters(), lr=2e-5)
    model.train()
    for epoch in range(2):  # 2 epochs
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Evaluate BERT
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions) * 100
    precision = precision_score(true_labels, predictions, average="binary", zero_division=1) * 100
    recall = recall_score(true_labels, predictions, average="binary", zero_division=1) * 100
    f1 = f1_score(true_labels, predictions, average="binary", zero_division=1) * 100

    print(f"BERT - Hindi Hate Speech: Accuracy={accuracy:.2f}%, Precision={precision:.2f}%, Recall={recall:.2f}%, F1={f1:.2f}%")
    return {
        "Model": "BERT",
        "Language": "Hindi",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

# Process the dataset
print("\nProcessing Hindi Hate Speech Dataset")
# Filter data
data["text"] = data["text"].fillna("").astype(str)
X = data["text"]
y = data["hate"].astype(int)

# Split data and reset indices
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

# Train and evaluate BERT
results = train_and_evaluate_bert(X_train, X_test, y_train, y_test)
all_results = [results]

# Save results to a CSV file
if all_results:
    results_df = pd.DataFrame(all_results)
    results_file = os.path.join(base_dir, "bert_hindi_hate_performance.csv")
    results_df["Accuracy"] = results_df["Accuracy"].map("{:.2f}%".format)
    results_df["Precision"] = results_df["Precision"].map("{:.2f}%".format)
    results_df["Recall"] = results_df["Recall"].map("{:.2f}%".format)
    results_df["F1-Score"] = results_df["F1-Score"].map("{:.2f}%".format)
    results_df.to_csv(results_file, index=False)
    print(f"\nSaved BERT performance to {results_file}")
else:
    print("\nNo results to save due to errors or empty data.")