# import model1, model2
import warnings

import model3
from data_prep import make_dataframe
from data_processing import (
    add_sentiment_and_category_columns,
    make_pivots,
    make_summary_strings,
    refine_categories,
)

warnings.filterwarnings("ignore")

# model1.main()                                              # Run Model 1 to get sentiment column (already stored in data/sentiment_columns.pkl)
# model2.main()                                              # Run Model 2 to get category column  (already stored in data/category_columns.pkl)

df_resampled = make_dataframe()  # Loads the dataset, cleans it, and reshufles the rows

df_full = add_sentiment_and_category_columns(
    df_resampled
)  # Adds sentiment and category columns to the dataframe

df_refined = refine_categories(
    df_full
)  # Refines the categories (all product IDs are assigned to the most common category)

best_products, worst_products = make_pivots(
    df_refined
)  # Filters for the best and worst products in each category


final_best_products = make_summary_strings(
    best_products, df_resampled, "best_products"
)  # Adds a column with a string of 20 reviews for the best products

final_worst_products = make_summary_strings(
    worst_products, df_resampled, "worst_products"
)  # Adds a column with a string of 20 reviews for the worst products

model3.main()  # Run Model 3 to generate an AI summary for the best and worst products, this is stored in data/best_products.json and data/worst_products.json
