import pandas as pd
import numpy as np
import kagglehub
import os
import random
import pickle
from pathlib import Path
import random
import warnings
warnings.filterwarnings("ignore")

def make_dataframe():

    # Download all files
    path = kagglehub.dataset_download("datafiniti/consumer-reviews-of-amazon-products")
    csv_paths = list(Path(path).glob("*.csv"))


    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)

    df = df.drop_duplicates(
        subset=['id', 'dateAdded', 'dateUpdated', 'reviews.text', 'reviews.title', 'reviews.username'],
        keep='first'
    ).reset_index(drop=True)

    df = df.dropna(subset=["reviews.text", "reviews.title", "reviews.rating"])

    df['name'] = df['name'].astype(str) 

    df['name_title_text'] = \
        "Name: " + df['name'].astype(str) + \
        " Title: " + df['reviews.title'].astype(str) + \
        " Review: " + df['reviews.text'].astype(str)



    df_resampled = df.sample(frac=1, random_state=42).reset_index(drop=True) #  randomize the rows of the dataframe
    df_resampled['new_id'] = np.arange(1, len(df_resampled) + 1)



    df_resampled['reviews.doRecommend'] = df_resampled['reviews.doRecommend'].replace({'true': 1, 'false': 0})
    df_resampled['reviews.didPurchase'] = df_resampled['reviews.didPurchase'].replace({'true': 1, 'false': 0})
    df_resampled['reviews.numHelpful'] = df_resampled['reviews.numHelpful'].replace({'true': 1, 'false': 0})
    df_resampled['reviews.rating'] = df_resampled['reviews.rating'].replace({'true': 1, 'false': 0})

    df_resampled['reviews.numHelpful'] = (df_resampled['reviews.numHelpful'].fillna(0)).astype(int)
    df_resampled['reviews.didPurchase'] = (df_resampled['reviews.didPurchase'].fillna(0)).astype(int)
    df_resampled['reviews.doRecommend'] = (df_resampled['reviews.doRecommend'].fillna(0)).astype(int)
    df_resampled['reviews.rating'] = (df_resampled['reviews.rating'].fillna(0)).astype(int)

    df_resampled = df_resampled[df_resampled['name'] != 'nan']

    return df_resampled

if __name__ == "__main__":
    df = make_dataframe()