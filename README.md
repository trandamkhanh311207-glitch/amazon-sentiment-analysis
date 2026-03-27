# Amazon Sentiment Analysis

## Executive Summary

* **Problem:** Build a sentiment classification system to identify negative feedback signals from Amazon product reviews.

* **Dataset:** Amazon Electronics reviews (~100K samples), converted into a binary sentiment classification task based on rating.

* **Best Model:** TF-IDF + Logistic Regression with class weighting and threshold optimization.

* **Key Insight:** High overall performance can hide critical failure cases, particularly in detecting minority-class (negative) reviews.

---

## 1. Problem Overview

Understanding customer sentiment is essential for product monitoring, quality control, and decision-making.

In real-world systems, detecting **negative feedback** is often more important than maximizing overall accuracy, since negative reviews highlight issues that require action.

This project focuses on building a sentiment classification pipeline while emphasizing:

* class imbalance
* evaluation beyond accuracy
* trade-offs between precision and recall

---

## 2. Dataset

* Source: Amazon Electronics Reviews
* Size: ~100,000 reviews
* Task: Binary classification

### Labeling Strategy

* Rating ≥ 4 → **Positive**
* Rating ≤ 2 → **Negative**
* Rating = 3 → removed (ambiguous)

### Additional Features

* Review length
* Word count
* Verified purchase flag

---

## 3. Methodology

### Text Processing

* Lowercasing
* Basic cleaning
* TF-IDF vectorization

### Models Evaluated

* Logistic Regression
* Naive Bayes

### Key Techniques

* **Class weighting** to address imbalance
* **Threshold tuning** to control decision behavior

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Precision-Recall Curve
* ROC Curve

---

## 4. Results

### Confusion Matrix

![Confusion Matrix](results/figures/confusion_matrix.png)

### Precision-Recall Curve

![PR Curve](results/figures/pr_curve.png)

### ROC Curve

![ROC Curve](results/figures/roc_curve.png)

---

### Threshold Optimization

![Threshold vs F1](results/figures/threshold_f1_curve.png)

Adjusting the decision threshold allows better control over precision-recall trade-offs:

* Lower thresholds improve **negative recall**
* Higher thresholds increase **precision**
* Optimal threshold depends on application needs, not just maximizing F1

---

## 5. Model Comparison

| Model                                 | Accuracy | Precision | Recall      | F1        | Notes                                    |
| ------------------------------------- | -------- | --------- | ----------- | --------- | ---------------------------------------- |
| Logistic Regression (baseline)        | 0.94     | ~0.95     | 0.50        | ~0.65     | High accuracy but poor negative recall   |
| Logistic Regression (class-weighted)  | 0.90     | ~0.88     | 0.88        | ~0.88     | Strong improvement in minority detection |
| Logistic Regression (threshold-tuned) | ~0.90    | varies    | up to ~0.90 | optimized | Flexible trade-off depending on use case |

These results show that optimizing for accuracy alone can be misleading in imbalanced classification tasks.

---

## 6. Error Analysis

The model performs well on clear and explicit sentiment expressions but struggles in more complex cases:

### Common Failure Cases

* **Mixed sentiment reviews**
  Example: “Works great, but the battery life is terrible.”

* **Negation handling**
  Example: “Not bad at all.”

* **Short or weak-signal reviews**
  Example: “OK”, “Fine”

* **Ambiguous language**
  Reviews with unclear or neutral tone

### Key Insight

TF-IDF relies on surface-level lexical features and does not capture:

* compositional meaning
* contextual dependencies
* sentiment reversal through negation

---

## 7. Data Insights

### Class Distribution

![Class Distribution](results/figures/class_distribution.png)

### Review Length Distribution

![Review Length](results/figures/review_length_distribution.png)

---

## 8. Key Takeaways

* High accuracy does not guarantee strong real-world performance
* Minority-class detection (negative reviews) is critical
* Threshold tuning is a **decision tool**, not just a metric optimization trick
* Classical models remain competitive when properly tuned

---

## 9. Limitations

* Labels are derived from ratings → potential label noise
* TF-IDF cannot capture deep semantic relationships
* No contextual understanding of language

---

## 10. Future Work

Potential improvements include:

* Using contextual embeddings (e.g., BERT) to capture semantic meaning
* More advanced error analysis techniques
* Domain-specific fine-tuning

---

## 11. Repository Structure

```
amazon-sentiment-analysis/
│
├── configs/
├── data/
│   └── processed/
├── demo/
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_data_cleaning_and_labeling.ipynb
│   ├── 03_eda.ipynb
│   └── 04_modeling_and_evaluation.ipynb
├── results/
│   ├── figures/
│   └── tables/
├── src/
├── tests/
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 12. How to Run

```bash
pip install -r requirements.txt
```

Open and run:

```
notebooks/main.ipynb
```

---

## 13. Conclusion

This project demonstrates that building an effective sentiment analysis system is not only about achieving high accuracy, but also about understanding model behavior under real-world constraints.

By focusing on class imbalance, threshold tuning, and error analysis, the project highlights the importance of aligning machine learning models with practical objectives.
