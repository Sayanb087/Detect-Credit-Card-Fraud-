# 💳 Credit Card Fraud Detection using Logistic Regression

## 🌐 Project Overview
This project applies Logistic Regression, a powerful statistical and machine learning technique, to detect fraudulent credit card transactions. The goal is to build a binary classification model that can distinguish between legitimate and fraudulent transactions based on transaction data.

## 📊 Dataset Overview
The dataset used for this task typically contains anonymized credit card transactions, where features have been transformed using PCA to protect confidentiality. It includes:
* Numerical Features (V1 to V28): Principal components derived from the original data.
* Amount: Transaction amount.
* Time: Time since the first transaction in the dataset.
* Class: Target label (1 = Fraud, 0 = Not Fraud).

## 🔍 Problem Statement
Given the imbalanced nature of fraud detection (frauds are rare), it is critical to:
* Apply proper feature scaling.
* Choose the right evaluation metrics (accuracy is not enough).
* Handle class imbalance carefully.

## 📚 Key Concepts Explained
### 🔢 Logistic Regression
A supervised learning algorithm used for binary classification. It predicts the probability of an instance belonging to a class using the logistic (sigmoid) function.

### Logistic Function (Sigmoid):
    P(Y=1|X) = 1 / (1 + e^-(b0 + b1X1 + b2X2 + ... + bnXn))
The output is between 0 and 1 and is interpreted as the probability of the positive class.

### 📈 Linear vs Logistic Regression
* Linear Regression: Used for predicting continuous values.
* Logistic Regression: Used for predicting probabilities and class labels.

### 🧠 SelectKBest with f_classif
* SelectKBest: Feature selection technique to retain the top k features based on a scoring function.
* f_classif: ANOVA F-value between label/feature. Higher score = more  discriminatory power.

### 🧪 StandardScaler
Standardizes features by removing the mean and scaling to unit variance:
      X_scaled = (X - mean) / std
Important for models like logistic regression to perform optimally.

### 🧰 Scikit-Learn (sklearn)
A popular Python library that provides simple and efficient tools for machine learning and statistical modeling:
* LogisticRegression
* SelectKBest
* classification_report
* confusion_matrix
* train_test_split

## 🛠️ Model Building Pipeline
* Import Libraries
* Load Dataset
* Preprocess Data
  * Handle imbalance
  * Feature selection with SelectKBest
  * Scale features using StandardScaler
* Train Logistic Regression Model
* Evaluate Model Performance

## 🧮 Evaluation Metrics

### ✅ Accuracy
Percentage of correctly predicted instances:
* Accuracy = (TP + TN) / (TP + TN + FP + FN)
### 📊 Confusion Matrix
Summarizes actual vs predicted outcomes:
* TP (True Positive): Fraud correctly identified
* TN (True Negative): Legitimate transaction correctly identified 
* FP (False Positive): Legitimate predicted as fraud
* FN (False Negative): Fraud missed

### 📋 Classification Report
Provides:
* Precision: TP / (TP + FP)
* Recall: TP / (TP + FN)
* F1 Score: Harmonic mean of precision and recall
* Support: Number of true instances for each label
  
      from sklearn.metrics import classification_report
      print(classification_report(y_test, y_pred))

## 🔍 Handling Imbalanced Dataset
Fraudulent transactions are rare. To address imbalance:
* Under-sampling: Remove some majority class samples.
* Over-sampling: Duplicate or synthetically generate minority class samples.
* Use precision, recall, F1 instead of just accuracy

## 🧪 Model Predictions
After training, predictions are made on the test set. Evaluation metrics are computed, and confusion matrix plotted.

## 🧾 Summary
This project provides an end-to-end demonstration of detecting fraudulent credit card transactions using logistic regression. It covers feature selection, scaling, modeling, and interpretation of results using confusion matrices and classification reports. Handling imbalanced data and evaluating with proper metrics ensures reliable performance insights.









