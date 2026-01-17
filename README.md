# Rumor Detection Project

## Overview
This project builds a machine learning model to classify tweets into categories based on their content. It uses the Rumor Detection ACL 2017 dataset, which contains tweets labelled as True, False, Non-Rumor and Unverified.

## Objectives
- Load and preprocess tweet data.
- Train a logistic regression model to classify tweets.
- Evaluate the model's performance.
- Deploy the model as a REST API using Flask.

## Data
The dataset comprises thousands of tweets annotated with one of four labels. The preprocessing pipeline includes tokenisation, stop-word removal and TF-IDF vectorisation.

## Model Architecture
The model is a standard logistic regression classifier trained on TF-IDF features. The pipeline is implemented using scikit-learn and can be replaced with more complex architectures if desired.

## Results
| Label        | Precision | Recall | F1-Score | Support |
|------------- |----------:|-------:|---------:|--------:|
| False        | 0.85      | 0.79   | 0.82     | 125     |
| Non-Rumor    | 0.73      | 0.87   | 0.79     | 103     |
| True         | 0.85      | 0.84   | 0.85     | 112     |
| Unverified   | 0.88      | 0.80   | 0.84     | 122     |
| **Overall**  | -         | -      | **0.82** | -       |

## Using the Model

### Training and Evaluation
Run `python main.py` to train the model and evaluate its performance. The script loads data, splits it into training and test sets, trains the classifier and prints the evaluation metrics.

### Serving Predictions
Start the Flask API by running:
```sh
python main.py
```
Then send a POST request to `http://127.0.0.1:5000/predict` with JSON payload:
```json
{"text": ["This is a new tweet to classify"]}
```
The API will return the predicted label.

## Conclusion
This project demonstrates that even simple machine learning models can effectively detect rumours on social media, with an overall accuracy of 82%. Future work could explore transformer-based models and integration with real-time feeds.

## Contact
For questions or feedback, please reach out via email listed in my GitHub profile.
