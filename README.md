# AI-Powered Fake News Detection

Complete end-to-end ML project for fake news classification.

## Setup in VS Code

1. Open folder in VS Code
2. Open terminal: `Ctrl + Shift + P` → "Terminal: Create New Terminal"
3. Run: `pip install -r requirements.txt`
4. Run: `python main.py`

## Output
- Visualizations → `outputs/images/`
- Best model → `outputs/models/`
- Comparison CSV → `outputs/model_comparison.csv`

- <img width="1224" height="1285" alt="image" src="https://github.com/user-attachments/assets/167f1d4b-d958-4cab-80a1-a18463fecc5f" />


# AI-Powered Fake News Detection

> An NLP-based machine learning system for classifying news articles as **Real** or **Fake** using text preprocessing, feature engineering, and multiple classification algorithms.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-purple)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-orange?logo=scikit-learn)

---

## Overview

This project implements an end-to-end **Fake News Detection pipeline** using Natural Language Processing (NLP) and Machine Learning.

The system processes raw news articles, converts textual information into numerical feature vectors using **Bag-of-Words and TF-IDF**, trains multiple classification models, and evaluates their performance using standard classification metrics.

### Pipeline

**News Data → Text Cleaning → Tokenization → Feature Extraction → Model Training → Prediction → Evaluation**

---

## Key Features

- Text preprocessing and normalization
- Stopword removal and tokenization
- Bag-of-Words feature extraction
- TF-IDF feature extraction
- Multiple ML classification algorithms
- Model performance comparison
- Confusion matrix analysis
- Accuracy, Precision, Recall and F1-Score evaluation
- Automated visualization generation
- Saved trained model and vectorizer
- CSV-based model comparison
- Modular and reproducible project structure

---

## Machine Learning Models

| Model | Type | Approach |
|---|---|---|
| KNN | Non-Parametric | Similarity-based classification |
| Logistic Regression | Parametric | Linear probabilistic classification |
| Random Forest | Ensemble | Multiple decision trees |
| MLP Neural Network | Deep Learning | Multi-layer neural network |

---

## NLP & Feature Engineering

### Text Preprocessing

The input articles undergo:

1. Lowercase conversion
2. Punctuation removal
3. Special-character removal
4. Number removal
5. Stopword removal
6. Tokenization
7. Stemming/Lemmatization

### Feature Extraction

Two primary text representation techniques are used:

- **Bag-of-Words (BoW)** – represents text based on word occurrence.
- **TF-IDF** – assigns importance to words based on their frequency across documents.

---

## Evaluation

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

The project also generates visual comparisons of model performance.

> **Note:** Results shown in this README should be replaced with the actual results generated from the dataset used.

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| KNN | — | — | — | — |
| Logistic Regression | — | — | — | — |
| Random Forest | — | — | — | — |
| MLP Neural Network | — | — | — | — |

---

## Sample Visualizations

### Model Performance

![Model Comparison](outputs/images/model_accuracy_comparison.png)

### Confusion Matrix

![Confusion Matrix](outputs/images/random_forest_confusion_matrix.png)

---

## Dataset

The project is designed for the **Fake News Detection Dataset** obtained from sources such as:

- Kaggle
- UCI Repository

The dataset should contain news text and corresponding labels indicating whether an article is **Real** or **Fake**.

Place the dataset files inside:

```text
dataset/
