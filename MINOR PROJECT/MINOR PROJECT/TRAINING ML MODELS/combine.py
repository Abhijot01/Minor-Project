import pandas as pd
import os

# Step 1: Define the base path and file names
base_path = r'C:\Users\Dell\Desktop\MINOR PROJECT\CSV Files'
dataset_files = [
    'cleaned_sentiments_preprocessed.csv',
    'cleaned_data4.csv',
    'hate_cleaned_output2.csv',
    'hindi_cleaned_output3.csv',
    'cleaned_output1.csv'
]

# Step 2: Load and combine the datasets
combined_data = []
for file_name in dataset_files:
    file_path = os.path.join(base_path, file_name)
    try:
        df = pd.read_csv(file_path)
        combined_data.append(df)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found. Please check the file path.")
        exit(1)

# Concatenate all datasets into a single DataFrame
combined_df = pd.concat(combined_data, axis=0, ignore_index=True)

# Step 3: Save the combined dataset to a new CSV file
output_path = os.path.join(base_path, 'COMBINED_SENTIMENTS.csv')
combined_df.to_csv(output_path, index=False)
print(f"Combined dataset saved successfully to {output_path}")