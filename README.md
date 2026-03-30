# Amazon Review Sentiment Analysis

An end-to-end NLP project on Amazon Electronics reviews that explores not only overall sentiment classification performance, but also **reliability under class imbalance, threshold optimization, slice-based evaluation, and manual error analysis**.

---

## Executive Summary

This project builds a lightweight sentiment classifier for Amazon Electronics reviews using **TF-IDF features** and **classical machine learning models**.

Rather than stopping at a single accuracy score, this project asks a more important question:

> **Can a simple, interpretable NLP pipeline remain highly effective on weakly labeled e-commerce review data, while still revealing where and why it fails?**

### Best Result
- **Best model:** Logistic Regression with class weighting + threshold tuning
- **Best threshold:** `0.16`
- **Best F1-score:** **0.9707**
- **Best accuracy:** **0.9462**

### Main Insight
A large portion of the final performance gain came **not from using a more complex model**, but from making **better optimization choices inside a lightweight pipeline** — especially:
- handling class imbalance
- tuning the decision threshold
- evaluating failure cases beyond aggregate metrics

This project shows that **high overall performance can still hide systematic weaknesses**, particularly on:
- negation-heavy reviews
- mixed-sentiment reviews
- ambiguous or context-dependent language

---

## Research Question

**How far can a lightweight and interpretable TF-IDF-based sentiment classifier be pushed on noisy e-commerce review data through class weighting, threshold tuning, and reliability-focused evaluation?**

This project is framed not just as a standard text classification task, but as an investigation into how strong aggregate metrics can still mask recurring linguistic failure modes.

---

## Dataset

This project uses a processed subset of the **Amazon Electronics review dataset**.

### Labeling Strategy
To create a cleaner binary sentiment task:
- **positive** = rating **4 or 5**
- **negative** = rating **1 or 2**
- **neutral reviews (rating = 3)** were excluded

### Why this matters
This labeling strategy makes the task practical and scalable, but it also introduces an important limitation:

> **Ratings are only a proxy for sentiment, not perfect sentiment annotations.**

That means some apparent model “errors” may actually reflect:
- label ambiguity
- mixed opinions inside the same review
- mismatch between rating and textual sentiment

---

## Project Structure

```text
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

Methodology
1. Text Processing

Reviews were cleaned and transformed into numerical representations using TF-IDF.

2. Models Evaluated

The project compares:

Logistic Regression (unweighted)
Logistic Regression with class_weight="balanced"
Multinomial Naive Bayes
3. Threshold Optimization

Instead of relying only on the default classification threshold of 0.5, the project searches for a threshold that maximizes F1-score.

4. Reliability-Focused Evaluation

In addition to standard metrics, this project includes:

model comparison
ablation study
slice-based evaluation
manual error analysis
Main Results
Model Comparison
Model	Setting	Accuracy	Precision	Recall	F1
Logistic Regression	balanced (threshold = 0.16)	0.9462	0.9590	0.9826	0.9707
Logistic Regression	unweighted (threshold = 0.5)	0.9449	0.9508	0.9905	0.9702
Multinomial Naive Bayes	default	0.9159	0.9158	0.9991	0.9556
Logistic Regression	balanced (threshold = 0.5)	0.9038	0.9860	0.9067	0.9447
Interpretation

The strongest configuration was Logistic Regression with class weighting and threshold tuning, but the comparison also revealed something important:

Naive Bayes achieved extremely high recall, but lower precision and weaker overall reliability
Unweighted Logistic Regression performed very strongly
Balanced Logistic Regression at the default threshold underperformed badly compared to its tuned version
This means threshold choice mattered enormously
Threshold Optimization

Threshold tuning was one of the most important parts of the project.

Key Result
Default threshold (0.5): F1 = 0.9447
Best threshold (0.16): F1 = 0.9707

This shows that the default classification threshold is not always optimal, especially in imbalanced sentiment tasks where recall and precision need to be balanced more carefully.

Why this matters

This is one of the central takeaways of the project:

A better optimization choice inside the same simple model can produce a larger improvement than switching to a different model family.

Ablation Study

To isolate where performance gains really came from, I ran a small ablation study.

Experiment	Setting	F1
Class weight	unweighted (threshold = 0.5)	0.9702
Class weight	balanced (threshold = 0.5)	0.9447
Threshold	balanced (threshold = 0.5)	0.9447
Threshold	balanced (threshold = 0.16)	0.9707
Ablation Takeaways
Threshold tuning produced the largest improvement
Class weighting changed the precision–recall trade-off substantially
Final gains came less from “bigger models” and more from careful evaluation and optimization choices
Slice-Based Evaluation

Aggregate metrics can hide systematic weaknesses, so I evaluated the best model on several linguistically meaningful subsets.

Slice	N	F1
short_reviews	6527	0.9866
negation_reviews	7868	0.9457
mixed_reviews	5499	0.9479
Interpretation
Short reviews

The model performs exceptionally well on short reviews.

This likely happens because short reviews often contain very direct sentiment cues such as:

“great”
“terrible”
“waste of money”
“worth it”
Negation-heavy reviews

Performance drops on negation-heavy reviews.

Examples:

“not good”
“doesn’t work”
“not worth the money”

This is a known weakness of bag-of-words style models, because TF-IDF does not truly understand sentence structure.

Mixed-sentiment reviews

Performance also drops on mixed reviews.

Examples:

“good sound but stopped working”
“nice design, but overpriced”
“works well, although battery life is poor”

These cases require the model to understand contrast and nuance, which is difficult for linear lexical features.

Main Slice Insight

While the model achieves very strong overall performance, slice-based evaluation shows that:

high aggregate metrics can still conceal consistent weaknesses on linguistically harder cases.

Error Analysis

To go beyond metrics, I manually reviewed misclassified examples and grouped them into recurring failure patterns.

Typical Error Categories

The manually annotated errors in this project clustered around patterns such as:

mixed sentiment
negation
context-dependent meaning
weak positive / weak negative signals
ambiguous sentiment
clear negative language that conflicted with weak labels
Why this matters

This turns the project from a standard benchmark into a more research-oriented investigation.

Instead of only reporting that the model made mistakes, the project identifies:

what kinds of mistakes happen
why they happen
which linguistic structures are especially difficult
Main Error Analysis Insight

Many apparent model failures are not random.
They arise from a small set of repeated linguistic and labeling issues:

weak supervision from ratings
compositional language
contrastive sentiment
ambiguous review wording
Calibration and Reliability

This project also includes a calibration analysis to examine whether the model’s predicted probabilities are trustworthy, not just whether the final class labels are correct.

This matters because:

a model can be accurate but poorly calibrated
confidence estimates matter in decision-sensitive settings
reliability is an important part of model quality, not just raw F1
Visual Outputs

The repository includes result artifacts such as:

class distribution
review length distribution
confusion matrix
precision-recall curve
ROC curve
threshold vs F1 plot
calibration plot
top positive terms
top negative terms
slice evaluation bar chart
error category distribution chart

These figures are stored in results/figures/.

Key Findings
1. Simple models can still be highly competitive

A lightweight TF-IDF + Logistic Regression pipeline achieved F1 ≈ 0.97, showing that interpretable classical models remain very strong on this task.

2. Optimization choices mattered more than complexity

The biggest gain did not come from changing model architecture, but from:

threshold tuning
reliability-focused evaluation
understanding the precision–recall trade-off
3. Aggregate metrics are not enough

Although the best model performed strongly overall, slice-based evaluation and manual error analysis revealed meaningful weaknesses on:

negation
mixed sentiment
ambiguous language
4. Weak labels matter

Because labels are derived from ratings rather than direct sentiment annotation, some “errors” reflect label noise or ambiguity, not just pure model weakness.

Limitations

This project intentionally focuses on classical, interpretable NLP models rather than large transformer-based systems.

Important limitations include:

labels are inferred from ratings rather than manually annotated sentiment
neutral reviews were excluded, making the task easier than fully realistic sentiment modeling
TF-IDF cannot fully capture context, negation scope, or compositional meaning
results are based on the Electronics subset and may not fully generalize to other product domains
some error categories were manually assigned on a sample, not on the entire test set
Future Work

Possible next steps include:

comparing against contextual embedding models
building a cleaner manually annotated benchmark subset
adding more robust calibration analysis
expanding slice evaluation to additional linguistic or product-specific subsets
exploring deployment via a lightweight interactive demo
Reproducibility
Run the notebook

Main notebook:

notebooks/main.ipynb
Install dependencies
pip install -r requirements.txt
Expected outputs

Running the notebook will generate:

results/tables/model_comparison.csv
results/tables/threshold_results.csv
results/tables/ablation_results.csv
results/tables/slice_evaluation.csv
figures in results/figures/
misclassified samples and annotation files in results/error_analysis/
Why this project is meaningful

This project is not just about building a sentiment classifier.

It is about showing that:

strong performance should be interpreted carefully
model evaluation should go beyond headline metrics
reliability and error understanding are essential parts of real-world NLP

In that sense, the project is closer to a mini research study on weakly labeled sentiment modeling than a basic machine learning exercise.
