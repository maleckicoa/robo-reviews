import torch
import pandas as pd
from pathlib import Path
from transformers import pipeline
import kagglehub
import pickle
from transformers import pipeline
import data_prep


print('\n\nModel 1 started\n\n')
df_resampled = data_prep.make_dataframe()

sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

def process_review(review):
    result = sentiment_model(review)[0]
    label = result["label"].lower()
    return label  # returns "positive", "negative", or "neutral"


def get_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return "neutral"

    result = sentiment_model(text[:512])[0]   # truncate very long reviews
    label = result["label"].lower()          # POSITIVE  NEGATIVE  NEUTRAL
    return label


df_resampled["full_review"] = (
    df_resampled["reviews.title"].astype(str) + " " +
    df_resampled["reviews.text"].astype(str)
).str.strip()


texts = (
    df_resampled["full_review"]
    .fillna("")
    .astype(str)
    .apply(lambda t: t[:1000])  # shorten if needed
    .tolist()
)

results = sentiment_model(texts, batch_size=16, padding=True)
df_resampled["sentiment"] = [r["label"].lower() for r in results]


# PICKLE SENTIMENT COLUMNS
sentiment_columns = df_resampled[['new_id', 'sentiment']]
with open(Path("data/sentiment_columns.pkl"), "wb") as f:
    pickle.dump(sentiment_columns, f)



print('\n\nModel 1 completed')