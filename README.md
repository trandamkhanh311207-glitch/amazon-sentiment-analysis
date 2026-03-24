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

## Model Comparison

To contextualize the performance of Logistic Regression, I trained a Multinomial Naive Bayes baseline on the same TF-IDF features.

| Model | Precision | Recall | F1-score |
|------|----------:|-------:|---------:|
| Logistic Regression | 0.986 | 0.907 | 0.945 |
| Multinomial Naive Bayes | 0.916 | 0.999 | 0.956 |

Although Multinomial Naive Bayes achieves a slightly higher F1-score, this is largely due to the strong class imbalance in the dataset. The model predicts the majority class very aggressively, which inflates recall but leads to poor minority-class behavior.

Logistic Regression is preferred because it provides more balanced and interpretable performance.

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
```
amazon-sentiment-analysis/
├── data/
├── notebooks/
│ └── main.ipynb
├── results/
├── README.md
└── requirements.txt
```

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
