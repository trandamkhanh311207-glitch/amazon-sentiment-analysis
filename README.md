# Amazon Review Sentiment Analysis
This project builds a sentiment analysis system to classify Amazon product reviews into positive and negative categories using natural language processing (NLP) techniques.

## Objective
The goal of this project is to:
- Analyze customer reviews from Amazon
- Identify patterns in user sentiment
- Build a classification model to detect negative feedback
- Handle real-world challenges such as class imbalance

## Dataset
The dataset is a subset of Amazon product reviews (Electronics category), containing:
- Review text
- Star ratings
- Metadata such as votes and verified purchase status
A sample of approximately 100,000 reviews is used for efficient processing.

## Methodology
The project follows a structured machine learning pipeline:
1. Data Loading  
   - Read compressed JSON data and convert to structured format  
2. Data Cleaning  
   - Handle missing values  
   - Normalize numeric fields  
   - Select relevant features  
3. Labeling  
   - Ratings ≥ 4 → Positive  
   - Ratings ≤ 2 → Negative  
   - Neutral reviews removed  
4. Exploratory Data Analysis (EDA)  
   - Analyze review length, helpful votes, and verified purchases  
5. Modeling  
   - TF-IDF vectorization  
   - Logistic Regression baseline  
6. Handling Class Imbalance  
   - Apply class weighting  
7. Threshold Tuning  
   - Adjust decision threshold to improve recall  
8. Evaluation  
   - Precision, Recall, F1-score  
   - Confusion Matrix


## Results
- Baseline model achieved high accuracy but failed to detect negative reviews effectively.
- After applying class weighting:
  - Recall for negative reviews improved significantly
- Threshold tuning further improved model flexibility.
Final model:
- Achieves strong performance on positive reviews
- Significantly improves detection of negative feedback

## Key Performance
- Accuracy: ~90%  
- Negative Recall: ~0.88  
- Positive F1-score: ~0.94  
The model is optimized to detect negative reviews, which are critical in real-world monitoring systems.

## Key Insights
- Negative reviews tend to be longer and more detailed
- Negative reviews receive more helpful votes
- Verified purchases are more likely to be positive
These insights highlight behavioral patterns in user-generated content.

## Real-World Impact
Understanding customer sentiment is critical for businesses to:
- Detect negative feedback early  
- Improve product quality  
- Enhance customer satisfaction  
This project demonstrates how machine learning can be applied to large-scale user-generated data to extract actionable insights.

## Limitations
- The model may struggle with sarcasm or complex language
- Class imbalance still affects precision
- Only basic NLP techniques (TF-IDF) are used
Future improvements could include deep learning models such as BERT.

## Conclusion
This project demonstrates how to build a practical sentiment analysis system while addressing real-world challenges such as class imbalance and evaluation trade-offs.
It emphasizes not only model performance but also interpretability and decision-making.

## Future Work
- Experiment with advanced models (e.g., BERT)
- Improve text preprocessing
- Deploy as an API or web application
