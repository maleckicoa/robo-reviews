import pickle
from pathlib import Path


def add_sentiment_and_category_columns(df_resampled):
    """
    Add sentiment and category columns to the dataframe
    """
    with open(Path("data/sentiment_columns.pkl"), "rb") as f:
        sentiment_columns = pickle.load(f)

    with open(Path("data/category_columns.pkl"), "rb") as f:
        category_columns = pickle.load(f)

    # with open("data/embedding_columns_5.pkl", "rb") as f:
    # embedding_columns = pickle.load(f)

    df_resampled = df_resampled.merge(sentiment_columns, on="new_id", how="left")
    df_resampled = df_resampled.merge(category_columns, on="new_id", how="left")
    # df_resampled = df_resampled.merge(embedding_columns, on="new_id", how="left")

    return df_resampled


def refine_categories(df_resampled):
    """
    Refine the categories by picking the most common category for each product
    """

    df_major_category = (
        df_resampled.groupby(["id", "predicted_product_category"])
        .size()
        .reset_index(name="count")
    )

    # Pick the category with the highest count for each id
    df_major_category = (
        df_major_category.sort_values(["id", "count"], ascending=[True, False])
        .groupby("id")
        .first()
        .reset_index()
        .rename(columns={"predicted_product_category": "final_category"})
    )

    df_resampled = df_resampled.merge(
        df_major_category[["id", "final_category"]], on="id", how="left"
    )
    df_resampled["predicted_product_category"] = df_resampled["final_category"]

    return df_resampled


def make_pivots(df_resampled):
    """
    Make the pivots for the best and worst products
    """
    out = (
        df_resampled.groupby(["predicted_product_category", "id"])
        .agg(
            # existing
            count_new_id=("new_id", "count"),
            count_positive=("sentiment", lambda x: (x == "positive").sum()),
            count_negative=("sentiment", lambda x: (x == "negative").sum()),
            count_neutral=("sentiment", lambda x: (x == "neutral").sum()),
            # new fields
            count_didPurchase=("reviews.didPurchase", "sum"),
            count_doRecommend=("reviews.doRecommend", "sum"),
            sum_numHelpful=("reviews.numHelpful", "sum"),
            sum_rating=("reviews.rating", "mean"),
            # name_first = ("name", "first"),
            name=("name", lambda x: x.value_counts().idxmax()),
            imageURLs=(
                "imageURLs",
                lambda x: x.dropna().value_counts().idxmax()
                if x.dropna().size > 0
                else None,
            ),
        )
        .reset_index()
    )

    out = out[(out["count_new_id"] >= 5)].reset_index(drop=True)

    out["count_new_id_by_category"] = out.groupby("predicted_product_category")[
        "count_new_id"
    ].transform("sum")

    out["count_do_recommend_by_category"] = (
        out.groupby("predicted_product_category")["count_doRecommend"].transform("sum")
    ).replace(0, 1)

    out["positive_sentiment_ratio"] = out["count_positive"] / out["count_new_id"]
    out["negative_sentiment_ratio"] = out["count_negative"] / out["count_new_id"]
    out["neutral_sentiment_ratio"] = out["count_neutral"] / out["count_new_id"]

    out["sentiment_score"] = (
        out["positive_sentiment_ratio"] - out["negative_sentiment_ratio"]
    )
    out["rating_score"] = out["sum_rating"] / 5

    out["frequency_score"] = out["count_new_id"] / out["count_new_id_by_category"]

    out["recommendation_score"] = (
        out["count_doRecommend"] / out["count_do_recommend_by_category"]
    )

    out["total_score_1"] = (
        0.35 * out["sentiment_score"]
        + 0.35 * out["rating_score"]
        + 0.15 * out["frequency_score"]
        + 0.15 * out["recommendation_score"]
    )
    out["total_score_2"] = (
        0.4 * out["sentiment_score"]
        + 0.4 * out["rating_score"]
        + 0 * out["frequency_score"]
        + 0.2 * out["recommendation_score"]
    )

    out["best_rank_in_category"] = (
        out.groupby("predicted_product_category")["total_score_1"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    out["worst_rank_in_category"] = (
        out.groupby("predicted_product_category")["total_score_1"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )

    front_cols = [
        "best_rank_in_category",
        "worst_rank_in_category",
        "total_score_1",
        "total_score_2",
    ]
    cols = front_cols + [col for col in out.columns if col not in front_cols]
    out = (
        out[cols]
        .sort_values(
            by=["predicted_product_category", "total_score_1"], ascending=False
        )
        .reset_index(drop=True)
    )
    out["category_id"] = out["predicted_product_category"] + "_" + out["id"]

    best_products = (
        out[out["best_rank_in_category"] <= 3]
        .sort_values(
            by=["predicted_product_category", "best_rank_in_category"], ascending=True
        )
        .reset_index(drop=True)
    )
    worst_products = out[out["worst_rank_in_category"] <= 3].sort_values(
        by=["predicted_product_category", "worst_rank_in_category"], ascending=True
    )
    worst_products = worst_products[
        ~worst_products["category_id"].isin(best_products["category_id"])
    ].reset_index(drop=True)

    return best_products, worst_products


def make_summary_strings(products_df, df_resampled, name="best_products"):
    """
    Select the 20 most helpful reviews for the best and worst products,
    join them into a single string and add it to the products dataframe
    """
    df_summary = df_resampled.copy()

    df_summary["category_id"] = (
        df_summary["predicted_product_category"] + "_" + df_summary["id"]
    )
    summary_df = df_summary[
        df_summary["category_id"].isin(products_df.category_id.unique())
    ]

    summary_strings = []

    for row in products_df.itertuples(index=False):
        col1 = row.category_id

        # Sentiment proportions
        col2 = row.positive_sentiment_ratio / (
            row.positive_sentiment_ratio + row.negative_sentiment_ratio
        )
        col3 = row.negative_sentiment_ratio / (
            row.positive_sentiment_ratio + row.negative_sentiment_ratio
        )

        # Get positive subset
        df_subset_pos = (
            summary_df.loc[
                (summary_df["category_id"] == col1)
                & (summary_df["sentiment"] == "positive")
            ]
            .sort_values(by="reviews.numHelpful", ascending=False)
            .head(int(20 * col2))
        )

        # Get negative subset
        df_subset_neg = (
            summary_df.loc[
                (summary_df["category_id"] == col1)
                & (summary_df["sentiment"] == "negative")
            ]
            .sort_values(by="reviews.numHelpful", ascending=False)
            .head(int(20 * col3))
        )

        # selected indexes
        selected_idx = list(df_subset_pos.index) + list(df_subset_neg.index)

        # extract & join text
        review_texts = summary_df.loc[selected_idx, "name_title_text"].tolist()
        summary_string = " ".join(review_texts)

        summary_strings.append(summary_string)

    products_df["summary_reviews_string"] = summary_strings

    cols = [
        "category_id",
        "predicted_product_category",
        "id",
        "count_new_id",
        "name",
        "imageURLs",
        "summary_reviews_string",
    ]
    products_df = products_df[cols]

    with open(Path(f"data/{name}.pkl"), "wb") as f:
        pickle.dump(products_df, f)

    return products_df
