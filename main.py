import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from flask import Flask, request, jsonify
import requests
import json

app = Flask(__name__)

# Use the correct dataset path
path = 'C:\\Users\\ayode\\.cache\\kagglehub\\datasets\\syntheticprogrammer\\rumor-detection-acl-2017\\versions\\1'
print("Path to dataset files:", path)

# List files in the download directory
files = os.listdir(path)
print("Files in download directory:", files)

# Initialize an empty DataFrame to hold combined data
combined_data = pd.DataFrame()

# Check for TXT files in subdirectories and combine data
for subdir in files:
    subdir_path = os.path.join(path, subdir)
    if os.path.isdir(subdir_path):
        subdir_files = os.listdir(subdir_path)
        print(f"Files in {subdir} directory:", subdir_files)
        
        # Load source_tweets.txt
        if 'source_tweets.txt' in subdir_files:
            tweets_path = os.path.join(subdir_path, 'source_tweets.txt')
            print(f"Loading tweets file: {tweets_path}")
            tweets = pd.read_csv(tweets_path, delimiter='\t', header=None, names=['tweet_id', 'text'])
            tweets['tweet_id'] = tweets['tweet_id'].astype(str)  # Convert tweet_id to string
            print("Tweets DataFrame:")
            print(tweets.head())
        
        # Load label.txt
        if 'label.txt' in subdir_files:
            labels_path = os.path.join(subdir_path, 'label.txt')
            print(f"Loading labels file: {labels_path}")
            labels = pd.read_csv(labels_path, delimiter='\t', header=None, names=['label_tweet_id'])
            labels[['label', 'tweet_id']] = labels['label_tweet_id'].str.split(':', expand=True)
            labels['tweet_id'] = labels['tweet_id'].astype(str)  # Convert tweet_id to string
            print("Labels DataFrame:")
            print(labels.head())
        
        # Merge tweets and labels
        if not tweets.empty and not labels.empty:
            data = pd.merge(tweets, labels[['tweet_id', 'label']], on='tweet_id')
            combined_data = pd.concat([combined_data, data], ignore_index=True)

# Inspect the combined data
print("Combined DataFrame:")
print(combined_data.head())

# Example preprocessing function
def preprocess_text(text):
    # Implement your preprocessing steps here
    return text.lower()

# Apply preprocessing
if not combined_data.empty:
    combined_data['processed_text'] = combined_data['text'].apply(preprocess_text)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(combined_data['processed_text'], combined_data['label'], test_size=0.2, random_state=42)

    # Convert text data to TF-IDF features
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Define the parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }

    # Initialize the GridSearchCV object
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')

    # Fit the grid search to the data
    grid_search.fit(X_train_tfidf, y_train)

    # Print the best parameters and best score
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)

    # Initialize and train the model with best parameters
    model = LogisticRegression(**grid_search.best_params_)
    model.fit(X_train_tfidf, y_train)

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)

    # Print the cross-validation scores
    print("Cross-validation scores:", cv_scores)
    print("Mean cross-validation score:", cv_scores.mean())

    # Make predictions
    y_pred = model.predict(X_test_tfidf)

    # Evaluate the model
    print(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(model, 'logistic_regression_model.pkl')

    # Save the TF-IDF vectorizer
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
else:
    print("No data available for training.")

# Load the model and vectorizer
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['text']
    data_processed = [preprocess_text(text) for text in data]
    data_tfidf = vectorizer.transform(data_processed)
    predictions = model.predict(data_tfidf)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
