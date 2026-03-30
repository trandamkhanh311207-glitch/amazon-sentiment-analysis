# Amazon Review Sentiment Analysis

This project explores sentiment classification on Amazon Electronics reviews using a lightweight NLP pipeline built with **TF-IDF** and **classical machine learning models**.

What I wanted to do here was not just train a model and report a high score. I also wanted to understand **why** the model performs well, **where** it struggles, and whether strong overall metrics still hide important failure cases.

That is why this project goes beyond standard model training and includes:
- threshold tuning
- model comparison
- slice-based evaluation
- manual error analysis

---

## Executive Summary

This project builds a binary sentiment classifier for Amazon Electronics reviews.

At first glance, it looks like a fairly standard NLP classification task. But the more interesting question behind the project was this:

> Can a simple, interpretable model still perform very strongly on noisy e-commerce review data, and if so, where does it remain unreliable?

The final best model was **Logistic Regression with class weighting and threshold tuning**, which achieved:

- **Best threshold:** `0.16`
- **Accuracy:** **0.9462**
- **F1-score:** **0.9707**

One of the main things I learned from this project is that the final improvement did not mainly come from choosing a more complicated model. A lot of the gain came from making better decisions **inside the same simple pipeline**, especially:
- tuning the classification threshold
- thinking carefully about class imbalance
- evaluating the model on harder subsets of language instead of relying only on one overall score

So in a way, this project became less about “finding the best model” and more about **understanding what good evaluation actually looks like**.

---

## Research Question

**How far can a lightweight and interpretable TF-IDF-based sentiment classifier be pushed on noisy e-commerce review data through threshold tuning, imbalance handling, and reliability-focused evaluation?**

I liked this framing because it shifts the project away from a simple benchmark mindset and toward a more honest question about performance and trustworthiness.

---

## Dataset

This project uses a processed subset of the **Amazon Electronics review dataset**.

To make the sentiment task cleaner, I converted ratings into binary labels:

- **positive** = rating **4 or 5**
- **negative** = rating **1 or 2**
- **neutral reviews (rating = 3)** were excluded

This makes the task easier to define, but it also introduces an important limitation:

> The labels are derived from ratings, not from direct human sentiment annotation.

That means some “mistakes” made by the model may not be true model failures at all. In some cases, the review text itself is mixed, ambiguous, or does not line up perfectly with the star rating.

That issue became especially obvious during the error analysis section.

---

## Project Structure

```
amazon-sentiment-analysis/
│
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── demo/
├── notebooks/
│   └── main.ipynb
├── results/
│   ├── error_analysis/
│   ├── figures/
│   └── tables/
├── src/
├── tests/
├── LICENSE
├── README.md
└── requirements.txt
```
## Methodology
1. Text Processing

The review text was cleaned and converted into numerical features using TF-IDF.

2. Models Evaluated

I compared three main setups:

Logistic Regression (unweighted)
Logistic Regression with class_weight="balanced"
Multinomial Naive Bayes
3. Threshold Optimization

Instead of relying only on the default threshold of 0.5, I searched for the threshold that produced the best F1-score.

4. Reliability-Focused Evaluation

Rather than stopping at a single classification report, I also included:

model comparison
ablation analysis
slice-based evaluation
manual error analysis

That part was important to me, because I did not want this project to turn into another “train model, print metrics, done” notebook.
## Main Results

### Model Comparison

| Model | Setting | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Logistic Regression | balanced (threshold = 0.16) | 0.9462 | 0.9590 | 0.9826 | **0.9707** |
| Logistic Regression | unweighted (threshold = 0.5) | 0.9449 | 0.9508 | 0.9905 | 0.9702 |
| Multinomial Naive Bayes | default | 0.9159 | 0.9158 | 0.9991 | 0.9556 |
| Logistic Regression | balanced (threshold = 0.5) | 0.9038 | 0.9860 | 0.9067 | 0.9447 |

### What stands out
A few things became clear from this comparison:

- **Logistic Regression with class weighting and threshold tuning** gave the strongest overall result.
- **Unweighted Logistic Regression** was already very competitive and performed surprisingly close to the best model.
- **Multinomial Naive Bayes** achieved extremely high recall, but weaker precision and lower overall balance.
- **Balanced Logistic Regression at the default threshold** underperformed quite a bit compared to its tuned version.

The most important takeaway here is that the biggest improvement did not come from switching to a completely different model. It came from making better choices inside the same simple pipeline, especially in how the decision threshold was handled.

## Threshold Optimization

This turned out to be one of the most important parts of the project.

### Key Result
Default threshold (0.5): F1 = 0.9447
Best threshold (0.16): F1 = 0.9707

That is a very large jump for a change that does not involve retraining a more advanced model.

### Why it matters

This was one of the clearest lessons in the project:

A smart change inside the same simple model can matter more than switching to a more complicated model.

That is exactly why I did not want to treat the default threshold as something fixed or unquestionable.

## Ablation Study

To understand where the final performance gains actually came from, I ran a small ablation analysis.

Instead of only reporting the best score, I wanted to isolate the effect of two key choices:
- class weighting
- threshold tuning

| Experiment | Setting | F1 |
|---|---|---:|
| Class weight | unweighted (threshold = 0.5) | 0.9702 |
| Class weight | balanced (threshold = 0.5) | 0.9447 |
| Threshold | balanced (threshold = 0.5) | 0.9447 |
| Threshold | balanced (threshold = 0.16) | **0.9707** |

### What this shows
A few things stand out from this table:

- **Threshold tuning** produced the biggest improvement.
- **Class weighting** clearly changed the precision–recall trade-off.
- The final gain came less from using a more advanced model and more from making better decisions within a lightweight pipeline.

That was one of the most interesting outcomes of the project for me. It showed that careful evaluation and optimization can matter just as much as model choice, and sometimes even more.

## Slice-Based Evaluation

A strong overall score does not always mean a model is equally reliable across all kinds of language.

To test this more carefully, I evaluated the best model on a few linguistically meaningful subsets of the test set:
- short reviews
- reviews containing negation
- reviews containing mixed-sentiment cues

| Slice | N | F1 |
|---|---:|---:|
| short_reviews | 6527 | **0.9866** |
| negation_reviews | 7868 | 0.9457 |
| mixed_reviews | 5499 | 0.9479 |

### Interpretation

#### Short reviews
The model performs extremely well on short reviews.

This makes sense because short reviews often contain very direct sentiment signals such as:
- “great”
- “terrible”
- “worth it”
- “waste of money”

Those are exactly the kinds of patterns a TF-IDF-based model can pick up easily.

#### Negation-heavy reviews
Performance drops on reviews with negation.

Examples:
- “not good”
- “doesn’t work”
- “not worth the money”

This is a common weakness of bag-of-words style models, since they can recognize important words but do not really understand how negation changes meaning.

#### Mixed-sentiment reviews
Performance also drops on reviews that contain both positive and negative signals.

Examples:
- “good sound but stopped working”
- “nice design, but overpriced”
- “works well, although battery life is poor”

These are harder because the model has to deal with contrast and nuance instead of just obvious positive or negative words.

### Main takeaway
Even though the model performs very strongly overall, slice-based evaluation shows that it is not equally reliable everywhere.

In particular, reviews with negation and mixed sentiment remain noticeably harder than short, direct reviews. That makes this part of the project especially important, because it shows where strong aggregate metrics can still hide real weaknesses.

## Error Analysis

To go beyond the headline metrics, I manually reviewed misclassified examples and grouped them into recurring error patterns.

### Common error categories
The annotated error cases in this project tended to fall into patterns such as:
- **mixed sentiment**
- **negation**
- **context-dependent meaning**
- **weak positive / weak negative signals**
- **ambiguous sentiment**
- **clear negative language paired with noisy labels**

### Why this matters
This part of the project changed the way I interpreted the results.

Before doing error analysis, it was easy to say:
> the model performs well overall

After reviewing the mistakes more closely, the conclusion became more honest:
> the model performs well overall, but it tends to fail in a few specific and repeated ways

That is much more useful than simply knowing the model made errors.

Instead of only reporting that mistakes exist, this section tries to explain:
- what kinds of mistakes happen
- why they happen
- which language patterns are especially difficult
- which cases may reflect noisy labels rather than pure model failure

### Main error analysis takeaway
Many of the errors were not random at all. They came from a relatively small number of repeated issues:
- noisy supervision from star ratings
- negation
- mixed sentiment in the same review
- vague or weak wording
- contextual meaning that TF-IDF cannot fully capture

That made the project feel less like a basic sentiment classification exercise and more like a small study of reliability under noisy labels.

---

## Calibration and Reliability

This project also includes a calibration analysis to check whether the model’s predicted probabilities are trustworthy, not just whether the final class labels are correct.

That matters because:
- a model can be accurate but still poorly calibrated
- confidence scores can matter a lot in practical settings
- reliability is part of model quality, not just raw F1

This section supports one of the main ideas behind the project:

> evaluation should not stop at one metric

A model can look strong on paper while still being less reliable than expected when its confidence estimates are examined more carefully.

---

## Visual Outputs

The repository includes a set of figures that help make the results easier to understand and interpret.

These include:
- class distribution
- review length distribution
- confusion matrix
- precision-recall curve
- ROC curve
- threshold vs F1 plot
- calibration plot
- top positive terms
- top negative terms
- slice evaluation bar chart
- error category distribution chart

All visual outputs are stored in `results/figures/`.

I wanted the project to be readable not only through tables and metrics, but also through figures that make the main patterns easier to see at a glance.

---

## Key Findings

### 1. Simple models can still go very far
A lightweight TF-IDF + Logistic Regression pipeline achieved **F1 ≈ 0.97**, which shows that classical models can still be highly competitive on this task.

### 2. Better evaluation mattered more than more complexity
The biggest improvement did not come from switching to a more advanced model. It came from:
- threshold tuning
- understanding trade-offs more carefully
- testing the model on harder subsets of language

### 3. A high overall score does not tell the whole story
Even with strong overall performance, the model still struggled more on:
- negation
- mixed sentiment
- ambiguous or context-heavy reviews

### 4. Weak labels really shape the results
Because labels come from ratings rather than direct sentiment annotation, some apparent “errors” are actually tied to ambiguity or label noise, not just poor modeling.

Overall, the project showed me that strong results are most convincing when they are paired with careful analysis of where the model is still weak.

---

## Limitations

This project intentionally focuses on classical, interpretable NLP models rather than large transformer-based systems.

Some important limitations are:
- labels are derived from ratings rather than manually annotated sentiment
- neutral reviews were excluded, which makes the task cleaner but also somewhat easier
- TF-IDF cannot fully capture context, negation scope, or compositional meaning
- results are based on the Electronics subset and may not generalize perfectly to other product domains
- manual error categories were assigned on a sample, not the entire dataset

I think it is important to be clear about these limitations, because a good project should be honest about what its results really mean.

---

## Future Work

There are several directions I would explore next if I continued this project:
- compare against contextual embedding models
- build a cleaner manually annotated benchmark subset
- expand slice evaluation to other linguistic patterns or domain-specific groups
- add more detailed calibration analysis
- turn the pipeline into a lightweight interactive demo

The current version already answers the main question I started with, but there is still plenty of room to push the project further in a more research-oriented direction.

## Reproducibility

If you want to reproduce the results in this project, the main entry point is the notebook below:

### Main notebook
```
notebooks/main.ipynb
```
### Install dependencies
```
pip install -r requirements.txt
```
### Expected outputs

Running the notebook will generate:
```
results/tables/model_comparison.csv
results/tables/threshold_results.csv
results/tables/ablation_results.csv
results/tables/slice_evaluation.csv
figures in results/figures/
misclassified samples and annotation files in results/error_analysis/
```

## Why this project matters to me
I wanted this project to be more than a standard sentiment classification notebook.

What interested me most was not just whether I could get a high score, but whether I could build a model that looked strong on paper and then still ask a harder question:

If a model performs extremely well overall, can we trust it equally across all kinds of language?

The answer here is clearly more nuanced than a single metric suggests.

The model is strong, but not equally strong everywhere.
That is exactly why this project includes threshold tuning, slice-based evaluation, and manual error analysis. Those parts made the project feel much more meaningful to me than simply reporting one final score.
