

# RoboReviews — The new product review aggregator

Product Review Aggregator Powered by NLP and LLMs

##  Executive Summary

RoboReviews transforms raw customer reviews into structured insights. It automates review ingestion, sentiment classification, product category grouping and article-style summary generation. The platform uses GPT Nano for category assignment and Llama for summary generation. Final product categories:
	•	BATTERIES
	•	E-READERS
	•	KIDS ELECTRONICS
	•	TABLETS
	•	STREAMINGING DEVICES

## Datasets 

Primary Dataset: https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products/data 

The system uses review title, review text, rating, brand, product metadata and category fields. All raw files are merged into a single cleaned dataframe.

### Automated Downloading from Kaggle

The project uses kagglehub to download and manage dataset files without manual intervention.

### Data Cleaning and Standardization

All processing is handled in data_prep.py, including:
- Merging raw datasets
- Normalizing column names
- Filtering corrupted rows
- Generating unique product identifiers
- Sampling for large datasets
- Pickle-based caching for faster reloads

## Methodology

###  Multiple Modeling Pipelines
main.ipynb allows you to:
- Load the cleaned dataset
- Explore review distributions
- Execute each model pipeline
- Inspect predictions and errors
- Export best or worst product picks as pickle or JSON files

### Model 1 — Sentiment Analysis

model1.py for sentiment classification using cardiffnlp/twitter-roberta-base-sentiment-latest (https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
Input: review title and full review text
Output: Positive, Neutral or Negative
Purpose: measure user satisfaction per review

### Model 2 — Category Classification with GPT Nano

model2.py for product category classification using gpt-4o-mini (https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/)
Input: product title, summary and aggregated review text
Output: one of the five final categories
Purpose: unify inconsistent product categories into a clean structure

### Model 3 — Generative Article Summary with Llama

model3.py for iterative article generation using meta-llama/Llama-3.2-1B-Instruct (https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
Top products are selected through an internal scoring formula combining sentiment distribution with number of reviews.
Llama generates a neutral summary of customer opinions and recommends the top products in each category.

## Requirements

- pandas 
- numpy
- kagglehub
- torch
- transformers
- accelerate
- jupyter
- pickle5
- sqlalchemy
- openai
- dotenv
- scikit-learn
- matplotlib

## Additional Notes

- All modeling components are modular and interchangeable.
- The system easily supports additional categories, larger datasets and different LLMs.
- The architecture is designed for experimentation, extension and fast iteration.
