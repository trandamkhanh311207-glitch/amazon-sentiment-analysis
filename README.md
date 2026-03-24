# 🧠 Amazon Review Sentiment Analysis

A machine learning project that classifies Amazon product reviews into **positive** and **negative** sentiments using NLP techniques.

This project goes beyond basic modeling by focusing on **model evaluation, threshold optimization, and error analysis** to understand real-world limitations of sentiment classification.

---

## 🎯 Objectives

- Build a complete NLP pipeline for sentiment classification
- Evaluate model performance using multiple metrics
- Optimize decision threshold using Precision-Recall trade-off
- Analyze model errors to uncover real-world challenges
- Compare different models to justify design choices

---

## 🏗️ Project Structure
```
amazon-sentiment-analysis/
│
├── data/ # Raw and processed data
├── notebooks/ # Jupyter notebooks
│ └── main.ipynb
├── results/ # Plots and evaluation outputs
├── README.md
└── requirements.txt
```

---

## ⚙️ Methodology

### 1. Data Processing
- Cleaned raw Amazon review data
- Removed missing and noisy samples
- Created sentiment labels based on rating:
  - Positive: rating ≥ 4
  - Negative: rating ≤ 2

---

### 2. Feature Engineering
- TF-IDF vectorization
- Sparse high-dimensional representation of text

---

### 3. Models

#### Logistic Regression (Main Model)
- Handles high-dimensional sparse data effectively
- Supports class weighting for imbalance handling

#### Multinomial Naive Bayes (Baseline)
- Fast and simple probabilistic model
- Strong baseline for text classification

---

## 📊 Model Performance

### Logistic Regression

| Metric | Score |
|------|------|
| Precision | 0.986 |
| Recall | 0.907 |
| F1-score | 0.945 |

---

### Multinomial Naive Bayes

| Metric | Score |
|------|------|
| Precision | 0.916 |
| Recall | 0.999 |
| F1-score | 0.956 |

---

## ⚖️ Model Comparison

Although Multinomial Naive Bayes achieves a slightly higher F1-score, a deeper analysis reveals important limitations.

- Naive Bayes achieves **near-perfect recall (≈1.00)** for the positive class
- However, it performs extremely poorly on the negative class (recall ≈ 0.11)

This happens due to **class imbalance**, where positive reviews dominate the dataset.

As a result:
- Naive Bayes predicts most samples as positive
- This inflates recall and F1-score artificially

👉 Logistic Regression is preferred because it provides:
- More balanced performance
- Better handling of minority class
- More reliable decision boundaries

---

## 🎯 Threshold Optimization

Instead of using the default threshold (0.5), I optimized the classification threshold using the Precision-Recall curve.

- Best threshold: **~0.16**
- Best F1-score: **~0.97**

This improves recall while maintaining strong precision, making the model more suitable for real-world applications.

---

## 📈 Precision-Recall Curve

The Precision-Recall curve was used to visualize the trade-off between precision and recall.

The optimal threshold was selected based on maximizing the F1-score.

---

## 🔍 Error Analysis

To better understand model limitations, I manually analyzed misclassified samples.

### Key Error Patterns

1. **Label Noise from Rating-Based Supervision (Most Frequent)**
   - Reviews labeled as positive but contain negative text (e.g., "poor quality")
   - Caused by mismatch between rating and actual sentiment

---

2. **Mixed Sentiment**
   - Reviews contain both positive and negative opinions
   - Difficult for bag-of-words models to determine dominant sentiment

---

3. **Context-Dependent Meaning**
   - Requires understanding sentence structure (e.g., "but", "however")
   - Not captured by TF-IDF features

---

4. **Weak Sentiment Signals**
   - Words like "good", "okay" lack strong polarity
   - Leads to ambiguity in classification

---

5. **Ambiguity**
   - Some reviews are inherently unclear

---

6. **Negation**
   - Phrases like "not bad" are misinterpreted
   - TF-IDF cannot model word interactions

---

## 🧠 Key Insights

- High accuracy does not guarantee real-world reliability
- Class imbalance can heavily bias model behavior
- Threshold tuning significantly impacts performance
- Error analysis is essential for understanding model limitations

---

## 🚀 Future Improvements

- Use advanced models (e.g., BERT, RoBERTa)
- Handle negation and context more effectively
- Improve labeling strategy (reduce noise)
- Perform hyperparameter tuning
- Deploy as a real-time sentiment analysis API

---

## 🛠️ Installation

```
bash
git clone https://github.com/trandamkhanh311207-glitch/amazon-sentiment-analysis.git
cd amazon-sentiment-analysis
pip install -r requirements.txt
