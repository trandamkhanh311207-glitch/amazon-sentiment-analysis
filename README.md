# 📊 Amazon Review Sentiment Analysis

## 🔍 Overview

This project builds a complete sentiment analysis pipeline on Amazon product reviews using Natural Language Processing (NLP) techniques.

The goal is not only to classify reviews as positive or negative, but also to **analyze model behavior, understand its limitations, and extract meaningful insights from errors**.

---

## 🎯 Objectives

- Build a full NLP pipeline from raw text to predictions
- Evaluate model performance using multiple metrics
- Analyze classification errors to understand model limitations
- Explore threshold optimization using Precision–Recall trade-offs

---

## 🧠 Methodology

### 1. Data Processing
- Cleaned raw review text
- Removed noise (punctuation, lowercasing, etc.)
- Labeled sentiment based on rating (positive vs negative)

### 2. Feature Engineering
- Used **TF-IDF vectorization**
- Converted text into numerical features

### 3. Modeling
- Logistic Regression (main model)
- Multinomial Naive Bayes (baseline)

### 4. Evaluation
- Precision, Recall, F1-score
- Confusion Matrix
- Threshold analysis

---

## 📈 Model Performance

| Model | Precision | Recall | F1-score |
|------|----------|--------|----------|
| Logistic Regression | ~0.986 | ~0.907 | ~0.944 |
| Naive Bayes | ~0.916 | ~1.0 | ~0.955 |

Naive Bayes achieves higher recall but is biased toward the majority class, while Logistic Regression provides more balanced performance.

---

## ⚖️ Threshold Optimization

Instead of using the default threshold (0.5), the optimal threshold was found using the Precision–Recall curve.

- **Best threshold:** ~0.16
- Improves recall significantly
- Demonstrates trade-off between precision and recall

📌 Key insight:
> Model performance is not fixed — it depends heavily on decision threshold.

---

## 🔬 Error Analysis

A detailed error analysis was conducted by manually inspecting misclassified samples.

### Key Findings:

1. **Label Noise**
   - Some reviews contain negative text but are labeled positive due to rating-based labeling
   - Indicates weak supervision in the dataset

2. **Mixed Sentiment**
   - Reviews often contain both positive and negative opinions
   - Difficult for linear models to classify correctly

3. **Context Dependency**
   - Words like "but", "however" require understanding sentence structure

4. **Negation Handling**
   - Phrases like "not bad" are misinterpreted

5. **Weak Sentiment Signals**
   - Words like "good", "okay" provide weak signals

---

## 📊 Precision–Recall Curve

The Precision–Recall curve highlights the trade-off between precision and recall.

The optimal operating point shows that:
- Lower thresholds increase recall
- Higher thresholds increase precision

---

## 🧠 Key Insights

- Model performance is influenced by both **algorithm choice** and **data quality**
- TF-IDF + Logistic Regression performs well on explicit sentiment
- However, it struggles with:
  - context
  - negation
  - subtle sentiment
- Threshold tuning can significantly improve performance without changing the model

---

## 🏗️ Project Structure
amazon-sentiment-analysis/
│
├── data/
├── notebooks/
│ └── main.ipynb
├── results/
├── README.md
└── requirements.txt

---

## 🚀 Future Improvements

- Use deep learning models (LSTM, BERT)
- Incorporate word embeddings
- Improve labeling quality
- Perform cross-validation
- Deploy as a web application

---

## 🧑‍💻 Author

**Tran Dam Khanh**

---

## ⭐ Why This Project Matters

This project goes beyond basic classification by focusing on:

- understanding model behavior
- identifying real-world data issues
- making informed decisions using evaluation metrics

It reflects a strong foundation in both **machine learning and analytical thinking**.
