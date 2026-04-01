<div align="center">

# 🕵️ Rumor Detection in Social Media

### Multi-Class Tweet Classification with Machine Learning + Flask API

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Flask](https://img.shields.io/badge/Flask-API-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![NLP](https://img.shields.io/badge/NLP-TF--IDF-8A2BE2?style=for-the-badge)](#)
[![Status](https://img.shields.io/badge/Status-Completed-16a34a?style=for-the-badge)](#)

<br/>

> **End-to-end rumor detection pipeline** for classifying tweets into  
> **True**, **False**, **Non-Rumor**, and **Unverified** categories.
>
> Built with classical NLP + Logistic Regression and deployed as a REST API.

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Objectives](#-objectives)
- [Dataset](#-dataset)
- [Pipeline](#-pipeline)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [API Usage](#-api-usage)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🔍 Overview

This project applies supervised machine learning to detect rumor-related signals in social media text.  
Using the **Rumor Detection ACL 2017 dataset**, tweets are classified into four categories:

- ✅ **True**
- ❌ **False**
- 🟦 **Non-Rumor**
- ❓ **Unverified**

Despite the complexity of online misinformation, a well-tuned traditional ML approach (TF-IDF + Logistic Regression) delivers strong and interpretable performance.

---

## 🎯 Objectives

- Load and preprocess tweet text data
- Build a robust NLP feature pipeline
- Train a multi-class classifier
- Evaluate performance with precision, recall, and F1-score
- Serve model predictions through a Flask REST API

---

## 📦 Dataset

| Field | Details |
|---|---|
| **Dataset** | Rumor Detection ACL 2017 |
| **Task** | Multi-class tweet classification |
| **Classes** | True, False, Non-Rumor, Unverified |
| **Input** | Raw tweet text |
| **Output** | Predicted rumor category |

### Preprocessing Steps

- Text cleaning and normalization
- Tokenisation
- Stop-word removal
- TF-IDF vectorisation

---

## ⚙️ Pipeline

```text
Raw Tweets
   ↓
Text Preprocessing
   ↓
TF-IDF Vectorizer
   ↓
Logistic Regression Classifier
   ↓
Evaluation + Flask Inference API
```

---

## 🧠 Model Architecture

The baseline model is a **Logistic Regression** classifier trained on **TF-IDF** features using `scikit-learn`.

### Why this setup?

- Strong baseline for sparse text representations
- Fast training and inference
- Interpretable coefficients
- Easy deployment and maintenance

---

## 📊 Results

| Label | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| **False** | 0.85 | 0.79 | 0.82 | 125 |
| **Non-Rumor** | 0.73 | 0.87 | 0.79 | 103 |
| **True** | 0.85 | 0.84 | 0.85 | 112 |
| **Unverified** | 0.88 | 0.80 | 0.84 | 122 |
| **Overall (Macro/Weighted)** | - | - | **0.82** | - |

> ✅ **Overall F1-score: 0.82**  
> This confirms that even a simple linear model can perform competitively on rumor detection tasks.

---

## 🚀 API Usage

### 1) Train + Evaluate Model

```bash
python main.py
```

This script:
- Loads and preprocesses data
- Splits train/test sets
- Trains the classifier
- Prints evaluation report
- Starts Flask app (if configured in same script)

### 2) Start Prediction API

```bash
python main.py
```

Default endpoint:

- `POST http://127.0.0.1:5000/predict`

### Example Request

```json
{
  "text": ["This is a new tweet to classify"]
}
```

### Example Response

```json
{
  "prediction": ["Unverified"]
}
```

---

## 📁 Project Structure

```text
📦 rumor-detection-project
 ┣ 📄 main.py                  ← Training + evaluation + Flask app
 ┣ 📄 README.md                ← Project documentation
 ┣ 📂 data/                    ← Dataset files
 ┣ 📂 models/                  ← Saved model/vectorizer (optional)
 ┗ 📂 notebooks/               ← Experiments/EDA (optional)
```

---

## 🛠 Tech Stack

```python
# Core
Python, pandas, numpy

# NLP + ML
scikit-learn (TF-IDF, LogisticRegression)

# API
Flask

# Utilities
joblib / pickle (model persistence)
```

---

## 🔮 Future Improvements

- Fine-tune transformer models (BERT, RoBERTa)
- Handle class imbalance with weighted loss / augmentation
- Add confidence scores and explainability layer
- Integrate real-time tweet stream processing
- Dockerize API for production deployment

---

## 👤 Author

<div align="center">

**Ayodele Isaiah Idowu**  
MSc Applied Economics · Development & Agricultural Economics  
University of Göttingen · DAAD LfA Scholar

[![Website](https://img.shields.io/badge/Portfolio-ayodeleid.com-0A66C2?style=for-the-badge&logo=google-chrome&logoColor=white)](https://ayodeleid.com)
[![GitHub](https://img.shields.io/badge/GitHub-AyodeleID-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AyodeleID)

</div>
