# Spam-SMS-Detection

## Overview
This project involves building an AI model to classify SMS messages as either spam or legitimate (ham). The goal is to preprocess the text data, extract meaningful features using techniques like TF-IDF or word embeddings, and then use machine learning algorithms such as Naive Bayes, Logistic Regression, or Support Vector Machines (SVM) to classify messages. The model's performance will be evaluated using various metrics such as accuracy, precision, recall, and F1-score.

## Technologies Used
* Python: The programming language used for implementation.
* scikit-learn: For machine learning algorithms and model evaluation.
* nltk: For natural language processing (NLP) tasks.
* pandas: For data manipulation and analysis.
* NumPy: For numerical computations.
## Dataset
The project uses a labeled dataset of SMS messages where each message is marked as either spam or ham. You can download datasets like the SMS Spam Collection Dataset from various sources (e.g., Kaggle or UCI repository).

# Steps Taken
## 1. Data Preprocessing
* Text cleaning: We clean the text by removing unwanted characters, digits, punctuation, and stopwords.
* Tokenization: Breaking the text into individual words (tokens).
* Feature Extraction: Using TF-IDF (Term Frequency-Inverse Document Frequency) to convert the text into numerical features. Alternatively, word embeddings like Word2Vec or GloVe can be used for better semantic understanding.
## 2. Model Training
* Naive Bayes: We implement the Multinomial Naive Bayes classifier, which is commonly used for text classification tasks.
* Logistic Regression: A linear model that works well for binary classification tasks.
* Support Vector Machine (SVM): A powerful classifier that can be used for text classification, especially with high-dimensional data.
## 3. Model Evaluation
The model is evaluated using the following metrics:
* Accuracy: The proportion of correctly classified messages.
* Precision: The proportion of spam messages correctly identified.
* Recall: The ability of the model to detect all spam messages.
* F1-Score: The harmonic mean of precision and recall, useful when dealing with imbalanced datasets.
