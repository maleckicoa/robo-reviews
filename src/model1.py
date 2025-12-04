import pickle
from pathlib import Path

import pandas as pd
from transformers import pipeline

import data_prep


def main(
    output_path: str = "data/sentiment_columns.pkl",
    batch_size: int = 16,
    max_text_len: int = 1000,
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
) -> pd.DataFrame:
    """
    Run sentiment analysis over reviews and stores only the the subset dataframe with 'new_id' + 'sentiment' columns as  apickle file

    """

    print("\n\nModel 1 started\n\n")

    sp = pipeline("sentiment-analysis", model=model_name)
    df = data_prep.make_dataframe()

    df["full_review"] = (
        df["reviews.title"].astype(str) + " " + df["reviews.text"].astype(str)
    ).str.strip()
    texts = (
        df["full_review"]
        .fillna("")
        .astype(str)
        .apply(lambda t: t[:max_text_len])
        .tolist()
    )

    results = sp(texts, batch_size=batch_size, padding=True)
    df["sentiment"] = [r["label"].lower() for r in results]

    sentiment_df = df[["new_id", "sentiment"]]
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(sentiment_df, f)

    print("\n\nModel 1 completed\n")
    return sentiment_df


if __name__ == "__main__":
    main()
