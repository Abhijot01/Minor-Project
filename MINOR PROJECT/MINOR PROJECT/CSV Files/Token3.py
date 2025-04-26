import pandas as pd
import re

# ✅ Custom Hindi stopwords (expand as needed)
hindi_stopwords = set([
    'है', 'और', 'का', 'के', 'की', 'को', 'में', 'से', 'पर', 'यह', 'था', 'हैं',
    'एक', 'इस', 'कि', 'थे', 'नहीं', 'तो', 'भी', 'जब', 'तक', 'लेकिन', 'जो', 'हो', 'गया', 'कर', 'रहा', 'करना'
])

# ✅ Load data
df = pd.read_csv("hindi_cleaned.csv")
print("Columns in file:", df.columns)

# ✅ Text cleaning function
def clean_hindi_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^\u0900-\u097F\s]', '', text)  # Keep only Hindi characters
    text = re.sub(r'\s+', ' ', text).strip()        # Remove extra spaces
    words = text.split()
    filtered_words = [word for word in words if word not in hindi_stopwords]
    return ' '.join(filtered_words)

# ✅ Apply cleaning on correct column
df['cleaned_text'] = df['Post'].apply(clean_hindi_text)

# ✅ Save to CSV
df.to_csv("cleaned_output.csv", index=False)

print("✅ Cleaned file saved as cleaned_output.csv")
