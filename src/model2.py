import os
import random
import pickle
import numpy as np
from openai import OpenAI
from pathlib import Path
import time
from dotenv import load_dotenv, find_dotenv
from sklearn.metrics.pairwise import cosine_similarity

import data_prep 
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY  = os.getenv('OPENAI_API_KEY')

SET_CATEGORIES = 6


print('Model 2 started\n\n')
df_resampled = data_prep.make_dataframe()

client = OpenAI(
    api_key=OPENAI_API_KEY,
)

def get_completion(prompt, model="gpt-4o-mini", temperature=0):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    text = response.choices[0].message.content
    tokens_in = response.usage.prompt_tokens
    tokens_out = response.usage.completion_tokens
    tokens_total = response.usage.total_tokens
    
    return text, {"input": tokens_in, "output": tokens_out, "total": tokens_total}



# --- function to assign the most similar category
def assign_product_category(review_emb):
    # make sure we have a numpy array
    review_emb = np.array(review_emb).reshape(1, -1)

    scores = {}
    for p, emb in product_categories_embeddings.items():
        emb_2d = np.array(emb).reshape(1, -1)
        score = cosine_similarity(review_emb, emb_2d)[0][0]
        scores[p] = score

    return max(scores, key=scores.get)



def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embedding = np.array(response.data[0].embedding)
    usage = response.usage
    return embedding, usage

    

prompt_header = f"""
Determine {SET_CATEGORIES} product topics that are being discussed in the following text.
Make each item one or two words long.
Format your response as a list of items separated by commas, without numbering them.
Text: 
"""

prompt_categories_header = f"""
Based on the following text, determine {SET_CATEGORIES} product categories.
The categories must closely relate to categories that exist in typical retail E-commerce.
Format your response as a list of items separated by commas, without numbering them.
Text: 
"""

#########################################################################################
#########################################################################################

def get_product_categories(df, sample_size, cost_per_million_tokens):

    product_type_dict = {}
    start_index = 0
    sample_no = 1

    while start_index < len(df_resampled):
        end_index = min(start_index + sample_size, len(df_resampled))

        review_list = df_resampled['name_title_text'].iloc[start_index:end_index].tolist()
        review_text = "\n".join(review_list)
        prompt = f"{prompt_header}\n\n{review_text}"

        try:
            time.sleep(random.uniform(5, 10))
            text, usage = get_completion(prompt)
            product_type_dict[f"Categories: {sample_no}"] = text
            product_type_dict[f"Usage: {sample_no}"] = usage
            product_type_dict[f"Cost: {sample_no}"] = float(usage['total'])/1_000_000 * cost_per_million_tokens
            print(f"Categories {sample_no}:  {text}")
            print(f"Usage: {usage}")
            print(f"Cost: ${float(usage['total'])/1_000_000 * cost_per_million_tokens}")

        except Exception as e:
            print(f"Error: {e}")
            pass
        
        start_index+=sample_size
        sample_no+=1
        
    with open(Path("data/product_type_dict.pkl"), "wb") as f:
        pickle.dump(product_type_dict, f)



def make_embeddings(df_resampled, sample_size, cost_per_million_tokens):

    embeddings = []
    usage_dict = {}
    total_tokens = 0
    sample_no = 1

    for start in range(0, len(df_resampled), sample_size):
        end = start + sample_size
        batch = df_resampled["name_title_text"].iloc[start:end].tolist()

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )

        batch_embeddings = [np.array(item.embedding) for item in response.data]
        embeddings.extend(batch_embeddings)

        tokens = getattr(response.usage, "total_tokens", 0)
        total_tokens += tokens
        batch_cost = float(tokens) / 1_000_000 * cost_per_million_tokens

        # save to dict
        usage_dict[f"Usage {sample_no}"] = tokens
        usage_dict[f"Cost {sample_no}"] = round(batch_cost, 6)

        print(f"Processed {end} / {len(df_resampled)} rows")
        print(f"Total tokens: {total_tokens}")
        print(f"Cost: ${float(total_tokens)/1_000_000 * cost_per_million_tokens}")


    with open(Path("data/embeddings_cost_dict.pkl"), "wb") as f:
         pickle.dump(product_type_dict, f)

    return embeddings, usage_dict



#########################################################################################
#########################################################################################


# CREATE PRE-MADE CATEGORIES
get_product_categories(df_resampled, sample_size=1000, cost_per_million_tokens= 0.15)


# OPEN PRE-MADE CATEGORIES
try:
    with open("data/product_type_dict.pkl", "rb") as f:
        product_type_dict = pickle.load(f)
except Exception as e:
    print(f"Error: {e}")


# GET FINAL CATEGORIES FROM PRE-MADE CATEGORIES
all_categories = [v for k, v in product_type_dict.items() if k.startswith("Categories")]
categores_text = "\n".join(all_categories)
categores_prompt = f"{prompt_categories_header}\n\n{categores_text}"
text, usage = get_completion(categores_prompt)
product_categories = text.split(",")
print(product_categories)


# MAKE REVIEW EMBEDDINGS (for Name, Title, Review)
embeddings, usage_dict = make_embeddings(df_resampled, sample_size=100, cost_per_million_tokens=0.02)
df_resampled["review_embedding"] = embeddings


# MAKE PRODUCT CATEGORIES EMBEDDINGS
product_categories_embeddings = {}
for p in product_categories:
    emb, _ = get_embedding(p)
    product_categories_embeddings[p] = emb


# USE EMBEDDINGS TO PREDICT PRODUCT CATEGORIES
df_resampled["predicted_product_category"] = df_resampled["review_embedding"].apply(assign_product_category)


# PICKLE EMBEDDING COLUMNS
embedding_columns = df_resampled[['new_id', 'review_embedding']]
with open(Path("data/embedding_columns.pkl"), "wb") as f:
    pickle.dump(embedding_columns, f)


# PICKLE CATEGORY COLUMNS
category_columns = df_resampled[['new_id', 'predicted_product_category']]
with open(Path("data/category_columns.pkl"), "wb") as f:
    pickle.dump(category_columns, f)


print('\n\nModel 2 completed')