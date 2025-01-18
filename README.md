# Rumor Detection Project

## Project Overview
This project involves building a machine learning model to classify tweets into different categories based on their content. The dataset used for this project is the "Rumor Detection ACL 2017" dataset, which contains tweets and their corresponding labels indicating whether they are rumors or not.

## Objectives
- Load and preprocess tweet data.
- Train a logistic regression model to classify tweets.
- Evaluate the model's performance.
- Deploy the model as a REST API using Flask.

## Setup Instructions

### Prerequisites
- Python 3.11
- Required Python packages (listed in `requirements.txt`)

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/AyodeleID/Rumor-Detection-Project.git
   cd Rumor-Detection-Project

   Install the required packages:
pip install -r requirements.txt

Running the Code
Train the Model and Evaluate Performance
Run the main script to train the model and evaluate its performance:
python main.py

Serve the Model with Flask
Run the Flask API to serve the model:
python main.py

Making Predictions
You can make POST requests to the API to get predictions. Here is an example using Python's requests library:
import requests
import json

url = 'http://127.0.0.1:5000/predict'
data = {'text': ["This is a new tweet to classify"]}
headers = {'Content-Type': 'application/json'}

response = requests.post(url, data=json.dumps(data), headers=headers)
print(response.json())

Results and Analysis
The model achieved an overall accuracy of 82%. The detailed evaluation metrics are as follows:

False:

Precision: 0.85
Recall: 0.79
F1-Score: 0.82
Support: 125
Non-Rumor:

Precision: 0.73
Recall: 0.87
F1-Score: 0.79
Support: 103
True:

Precision: 0.85
Recall: 0.84
F1-Score: 0.85
Support: 112
Unverified:

Precision: 0.88
Recall: 0.80
F1-Score: 0.84
Support: 122
Overall Accuracy: 0.82

Conclusion
This project demonstrates the effectiveness of using machine learning for rumor detection on social media. The model can be used for various applications such as identifying misinformation, sentiment analysis, and content moderation.

Contact
For any questions or further information, please contact Ayodele Idowu at ayodele.idowuu@gmail.com.


