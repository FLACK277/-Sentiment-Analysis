Flipkart Laptop Reviews Sentiment Analysis
Overview
This repository contains code for analyzing sentiment in Flipkart laptop reviews using machine learning techniques. The implementation uses Natural Language Processing (NLP) to predict sentiment (positive/negative) based on review text and performs hyperparameter tuning to optimize model performance.
Dataset
The analysis uses the Flipkart Laptop Reviews dataset available on Kaggle. The dataset contains:

Product names
Overall ratings
Number of ratings and reviews
Individual review ratings
Review titles
Review text

Features

Comprehensive Text Preprocessing: Converts review text to lowercase, removes HTML tags, punctuation, stopwords, and performs lemmatization
Multiple ML Models: Implements and compares Support Vector Machine (SVM) and Random Forest classifiers
Hyperparameter Tuning: Uses GridSearchCV to find optimal hyperparameters for both models
Detailed Performance Analysis: Includes classification reports, confusion matrices, and model comparisons
Feature Importance Analysis: Identifies key words that influence sentiment predictions
Visual Analysis: Includes wordclouds and charts to visualize results

Requirements
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
wordcloud (optional, for generating word clouds)
Installation

Clone this repository:

git clone https://github.com/yourusername/flipkart-sentiment-analysis.git
cd flipkart-sentiment-analysis

Install required packages:

pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud

Download the dataset from Kaggle and place it in the project directory as flipkart_laptop_reviews.csv

Usage
Run the main script:
python sentiment_analysis.py
The script will:

Load and preprocess the dataset
Create and train ML models with hyperparameter tuning
Evaluate model performance
Generate visualizations
Save output files (confusion matrices, feature importance, etc.)

Files

sentiment_analysis.py: Main script containing all code for the analysis
svm_confusion_matrix.png: Confusion matrix for the SVM model
rf_confusion_matrix.png: Confusion matrix for the Random Forest model
model_comparison.png: Bar chart comparing model performances
feature_importance.png: Bar chart showing the most important features
sentiment_wordclouds.png: Word clouds for positive and negative sentiments

Methodology
Data Preparation

Creates a binary sentiment target (positive/negative) based on rating thresholds
Handles missing values in the dataset
Combines review title and text for richer feature extraction

Text Preprocessing
Model Training
The implementation builds two ML pipelines:

TF-IDF Vectorization + SVM
TF-IDF Vectorization + Random Forest

For each pipeline, GridSearchCV is used to find optimal hyperparameters like n-gram range, regularization parameters, kernel types, number of trees, etc.
Evaluation
Models are evaluated using:

Accuracy
Precision, Recall, F1-score
Confusion matrices
Analysis of misclassified examples

Results
The script outputs performance metrics for both models and identifies which words are most predictive of positive or negative sentiment. Detailed results will vary based on your specific run and hyperparameter search space.
Future Improvements

Implement more advanced NLP techniques like word embeddings (Word2Vec, GloVe)
Add deep learning models (LSTM, BERT)
Perform aspect-based sentiment analysis to identify specific laptop features mentioned positively or negatively
Add cross-validation for more robust hyperparameter tuning
Create an interactive dashboard for exploring the results

License
MIT License

Contributors
Pratyush Rawat

Acknowledgments
Dataset provided by Kaggle
NLTK for NLP tools and resources
