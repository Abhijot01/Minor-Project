import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import re

# Define file paths
base_path = r"C:\Users\Dell\Desktop\MINOR PROJECT\CSV Files"
model_path = os.path.join(base_path, "best_model_sentiment3.pkl")
vectorizer_path = os.path.join(base_path, "vectorizer3.pkl")
output_path = os.path.join(base_path, "predictions3.csv")

# Hardcoded example sentences from cleaned_data4.csv
example_data = [
    {
        "tweet": "These hoes ain't loyal ; no they ain't http://t.co/h1UBsVbkGl",
        "cleaned_text": "hoes ai loyal ai http smfh amp wonder nobody decent wants",
        "class": 1
    },
    {
        "tweet": "this is why i love birds. http://t.co/Gk2wiNhBkw",
        "cleaned_text": "love birds http",
        "class": 2
    },
    {
        "tweet": "ya bitch aint bad nigga, you aint got no taste",
        "cleaned_text": "ya bitch aint bad nigga aint got taste",
        "class": 1
    },
    {
        "tweet": "which one of these names is more offensive kike, wop, kraut, wetback jigaboo, towelhead, gook, or redskin.",
        "cleaned_text": "one names offensive kike wop kraut wetback jigaboo towelhead gook redskin",
        "class": 0
    },
    {
        "tweet": "this movie is actually good cuz its so retarded",
        "cleaned_text": "movie actually good cuz retarded",
        "class": 2
    },
    {
        "tweet": "u stupid bitch",
        "cleaned_text": "u stupid bitch",
        "class": 1
    },
    {
        "tweet": "wonder if they gon have em for the low on nigger friday....",
        "cleaned_text": "wonder gon em low nigger friday",
        "class": 0
    },
    {
        "tweet": "who's pussy better then a crazy bitch?",
        "cleaned_text": "pussy better crazy bitch",
        "class": 1
    },
    {
        "tweet": "~~Ruffled | Ntac Eileen Dahlia - Beautiful color combination of pink, orange, yellow &amp; white. A Coll http://t.co/H0dYEBvnZB",
        "cleaned_text": "ntac eileen dahlia beautiful color combination pink orange yellow amp white coll http",
        "class": 2
    },
    {
        "tweet": "you niggers cheat on ya gf's? smh....",
        "cleaned_text": "niggers cheat ya gf smh",
        "class": 1
    }
]

# Create DataFrame from example data
test_data = pd.DataFrame(example_data)

# Function to rephrase tweet into a full sentence
def rephrase_to_full_sentence(tweet, cleaned_text):
    # Remove URLs, special characters, and extra spaces
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
    tweet = re.sub(r"[^\w\s]", "", tweet)
    tweet = re.sub(r"\s+", " ", tweet).strip()
    
    # Simple heuristic to identify subject and context
    words = cleaned_text.lower().split()
    subject = "She" if any(word in words for word in ["bitch", "hoe", "girl", "pussy"]) else "He"
    verb = "is saying" if any(word in words for word in ["say", "said", "talk", "talking"]) else "is"
    
    # Handle common slang and abbreviations
    if "smh" in words:
        tweet += ", shaking her head"
    if "lol" in words:
        tweet += ", laughing out loud"
    
    # Construct full sentence
    full_sentence = f"{subject} {verb} {tweet.lower()}."
    return full_sentence.capitalize()

# Function to map class label to category name
def class_to_category(label):
    if label == 0:
        return "Hate Speech (Class 0)"
    elif label == 1:
        return "Offensive Language (Class 1)"
    else:
        return "Neither (Class 2)"

# Main function to run predictions and generate output
def run_predictions():
    try:
        # Extract test data
        X_test = test_data[['tweet', 'cleaned_text']]
        y_test = test_data['class']
        
        # Load model and vectorizer
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Transform test data
        X_test_vectorized = vectorizer.transform(X_test['cleaned_text'])
        
        # Make predictions
        y_pred = model.predict(X_test_vectorized)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100
        
        # Save predictions
        results = pd.DataFrame({
            'tweet': X_test['tweet'],
            'true_label': y_test,
            'predicted_label': y_pred
        })
        results.to_csv(output_path, index=False)
        
        # Print output
        print("Prediction Results for cleaned_data4.csv")
        print(f"The model is {accuracy:.2f}% accurate, which means it correctly identifies the category for {accuracy:.2f}% of the tweets in the test set.")
        print(f"The precision is {precision:.2f}%, which means when the model predicts a category, it is correct {precision:.2f}% of the time.")
        print(f"The recall is {recall:.2f}%, which means the model correctly finds {recall:.2f}% of all tweets that belong to each category.")
        print(f"The F1-score is {f1:.2f}%, which is a balanced measure combining precision and recall to show overall performance.")
        print()
        print(f"The predictions are saved in a file located at {output_path}.")
        print()
        print("Here are some sample predictions (first 10):")
        print()
        
        # Display all example predictions (up to 10)
        for i in range(min(10, len(X_test))):
            tweet = X_test['tweet'].iloc[i]
            cleaned_text = X_test['cleaned_text'].iloc[i]
            true_label = y_test.iloc[i]
            pred_label = y_pred[i]
            
            # Rephrase tweet
            full_sentence = rephrase_to_full_sentence(tweet, cleaned_text)
            
            print(f"Tweet: {full_sentence}")
            print(f"The true category is {class_to_category(true_label)}.")
            print(f"The model predicted this as {class_to_category(pred_label)}.")
            print()
        
        # Print class label meanings
        print("Meanings of the Class Labels:")
        print()
        print("Class 0 is called Hate Speech. This means a tweet is using harmful words or ideas to attack people because of their race, religion, or other personal traits. For example, a tweet might use rude slurs to insult a group, which is meant to hurt or spread hate. These tweets are very serious because they can make people feel unsafe or encourage others to dislike certain groups.")
        print()
        print("Class 1 is called Offensive Language. This means a tweet is using rude, vulgar, or insulting words, but it is not always targeting a specific group like Hate Speech does. For example, a tweet might include curse words or call someone a name to express anger or frustration. These tweets are inappropriate because they can upset people, but they are less harmful than Hate Speech.")
        print()
        print("Class 2 is called Neither. This means a tweet is safe and does not contain any rude or harmful language. For example, a tweet might talk about something positive, like liking birds, or just share a normal thought. These tweets are okay because they do not hurt anyone’s feelings or cause problems.")
    
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_predictions()