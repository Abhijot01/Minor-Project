import pandas as pd
import os
import re
import random

# Define paths
base_path = r'C:\Users\Dell\Desktop\MINOR PROJECT\CSV Files'
input_file = os.path.join(base_path, 'cleaned_sentiments_preprocessed.csv')

# Check if input file exists
if not os.path.exists(input_file):
    print(f"Error: File '{input_file}' not found in {base_path}")
    exit(1)

# Load original dataset
try:
    df = pd.read_csv(input_file)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# Verify required columns
required_columns = ['text', 'sentiment', 'cleaned_text']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Error: Missing columns: {missing_columns}")
    exit(1)

# Get unique sentiments
unique_sentiments = df['sentiment'].unique()
num_sentiments = len(unique_sentiments)
print(f"Found {num_sentiments} unique sentiments: {list(unique_sentiments)}")

# Calculate rows per sentiment for 10,000 new rows
total_new_rows = 10000
rows_per_sentiment = total_new_rows // num_sentiments
remainder = total_new_rows % num_sentiments

# Distribute rows equally
sentiment_counts = {sentiment: rows_per_sentiment for sentiment in unique_sentiments}
# Distribute remainder to first few sentiments
for i, sentiment in enumerate(unique_sentiments[:remainder]):
    sentiment_counts[sentiment] += 1

# Verify total
if sum(sentiment_counts.values()) != total_new_rows:
    print(f"Error: Total new rows ({sum(sentiment_counts.values())}) does not equal {total_new_rows}")
    exit(1)

# Function to preprocess text for cleaned_text column
def preprocess_text(text):
    # Remove emojis, special characters, punctuation; keep words, spaces, hashtags
    text = re.sub(r'[^\w\s#]', '', text)
    text = re.sub(r'\s+', ' ', text.lower().strip())
    return text

# Templates and components for text generation
activities = [
    'hiking', 'cooking', 'gaming', 'reading', 'painting', 'dancing', 'running', 'traveling',
    'shopping', 'studying', 'photography', 'gardening', 'yoga', 'biking', 'swimming',
    'writing', 'singing', 'exploring', 'meditating', 'baking'
]
locations = [
    'park', 'beach', 'city', 'forest', 'mountain', 'lake', 'home', 'cafe', 'museum',
    'market', 'library', 'gym', 'garden', 'concert', 'festival', 'bookstore', 'trail'
]
adjectives = [
    'beautiful', 'amazing', 'tough', 'exciting', 'peaceful', 'challenging', 'vibrant',
    'serene', 'disappointing', 'inspiring', 'lonely', 'joyful', 'stressful', 'wonderful'
]
hashtags = [
    '#LifeVibes', '#Mood', '#AdventureTime', '#ChasingDreams', '#GoodVibes', '#Struggle',
    '#NatureLover', '#CityLife', '#BookWorm', '#FitnessJourney', '#ArtLife', '#MusicLover'
]

# Sentiment-specific templates (generic to handle any sentiment)
templates = {
    'generic_positive': [
        "Enjoying a {adj} day {activity} at the {location}! {hashtag}",
        "Feeling great after a {adj} {activity} session! {hashtag}",
        "Loving this {adj} moment at the {location}! {hashtag}"
    ],
    'generic_happy': [
        "So happy to be {activity} with friends at the {location}! {hashtag}",
        "Laughter and joy while {activity} today! {hashtag}",
        "Pure happiness from a {adj} {activity} experience! {hashtag}"
    ],
    'generic_negative': [
        "Feeling down after a {adj} day at {location}.",
        "Struggling with {activity} today.",
        "Heart feels heavy after {adj} {activity}."
    ],
    'generic_neutral': [
        "Just another day {activity} at the {location}.",
        "Trying out {activity} today, nothing special.",
        "Spending time {activity} at {location}, feeling okay."
    ],
    'generic_excited': [
        "Can’t wait to go {activity} at the {location}! {hashtag}",
        "Super excited for {activity} this weekend! {hashtag}",
        "Thrilled about {adj} {activity} plans! {hashtag}"
    ]
}

# Map sentiments to template types
sentiment_template_map = {
    'positive': 'generic_positive',
    'happy': 'generic_happy',
    'excitement': 'generic_excited',
    'sad': 'generic_negative',
    'neutral': 'generic_neutral',
    'curiosity': 'generic_excited',
    'calm': 'generic_neutral',
    'hate': 'generic_negative',
    'fear': 'generic_negative',
    'awe': 'generic_positive',
    'love': 'generic_positive',
    'gratitude': 'generic_positive',
    'frustration': 'generic_negative',
    'nostalgia': 'generic_negative',
    'disappointment': 'generic_negative',
    'admiration': 'generic_positive',
    'anticipation': 'generic_excited',
    'pride': 'generic_positive',
    'acceptance': 'generic_neutral',
    'inspiration': 'generic_positive',
    'bitterness': 'generic_negative',
    'confusion': 'generic_negative',
    'elation': 'generic_happy',
    'hopeful': 'generic_positive',
    'empowerment': 'generic_positive'
}

# Generate new data
new_data = {'text': [], 'sentiment': [], 'cleaned_text': []}

# Create a list of sentiments based on counts
sentiments = []
for sentiment, count in sentiment_counts.items():
    sentiments.extend([sentiment] * count)
random.shuffle(sentiments)  # Shuffle to mix sentiments

# Generate rows
for sentiment in sentiments:
    # Select template type
    template_type = sentiment_template_map.get(sentiment, 'generic_neutral')
    template = random.choice(templates[template_type])
    # Fill template
    text = template.format(
        adj=random.choice(adjectives),
        activity=random.choice(activities),
        location=random.choice(locations),
        hashtag=random.choice(hashtags)
    )
    # Preprocess for cleaned_text
    cleaned_text = preprocess_text(text)
    # Append to new_data
    new_data['text'].append(text)
    new_data['sentiment'].append(sentiment)
    new_data['cleaned_text'].append(cleaned_text)

# Create DataFrame for new data
new_df = pd.DataFrame(new_data)

# Append new data to original dataset
augmented_df = pd.concat([df, new_df], ignore_index=True)

# Save augmented dataset to the same file
try:
    augmented_df.to_csv(input_file, index=False)
    print(f"Augmented dataset saved to {input_file}")
    print(f"Original rows: {len(df)}")
    print(f"New rows added: {len(new_df)}")
    print(f"Total rows: {len(augmented_df)}")
except Exception as e:
    print(f"Error saving augmented dataset: {e}")
    exit(1)

# Print sentiment distribution
print("\nSentiment Distribution in Augmented Dataset:")
sentiment_dist = augmented_df['sentiment'].value_counts().sort_index()
print(sentiment_dist)