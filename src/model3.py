import pandas as pd
import os
import pickle
from dotenv import load_dotenv, find_dotenv
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

import data_prep 


print('Model 3 started\n\n')


_ = load_dotenv(find_dotenv())
hf_token  = os.getenv('HF_TOKEN')

best_products = pd.read_pickle("data/best_products.pkl")
worst_products = pd.read_pickle("data/worst_products.pkl")


model_id = "meta-llama/Llama-3.2-1B-Instruct"

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=hf_token
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=hf_token,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

model.eval()





def llama_generate(system_prompt: str, user_prompt: str,
                   max_new_tokens: int = 220,
                   temperature: float = 0.2,
                   top_p: float = 0.9) -> str:
    prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.max_new_tokens = max_new_tokens
    gen_cfg.temperature = temperature
    gen_cfg.top_p = top_p
    gen_cfg.do_sample = False
    gen_cfg.pad_token_id = tokenizer.eos_token_id
    gen_cfg.repetition_penalty = 1.15          # push it away from repeating
    gen_cfg.no_repeat_ngram_size = 6           # avoid repeating long phrases

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            generation_config=gen_cfg,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    answer = full_text[len(prompt):].strip()
    return answer



def summarize_product_row_with_llama(row) -> str:
    product = row["name"]
    category = row.get("predicted_product_category", "")
    reviews = str(row["summary_reviews_string"])[:3500]

    system_prompt = (
        "You are a senior data analyst. "
        "Write a short, natural, factual paragraph that summarizes customer opinions. "
        "Keep it concise, avoid repetition, and avoid bullet points."
        "Include what customers appreciate and what they do not like. "
        "Avoid repetition and avoid promotional language."
    )

    user_prompt = (
        f"Product: {product}\n"
        f"Category: {category}\n\n"
        "Write one short paragraph describing how customers perceive this product. "
        "Use a neutral tone and avoid generic phrases such as "
        "'customers are satisfied', 'customers love', 'highly recommended', "
        "'excellent', 'great value', or any wording that exaggerates sentiment.\n\n"
        "Start with a neutral statement such as "
        "'Reviewers mention' or 'Users report'.\n\n"
        "Include both the most common positive themes and the most common drawbacks. "
        "Avoid praise that does not come from the reviews. "
        "Avoid softening negative points. "
        "No bullet points. Max 55 words.\n\n"
        "Example style:\n"
        "“Reviewers mention that the product performs reliably and is easy to use. "
        "They highlight its compact size and practical features. Some point out issues "
        "like inconsistent durability or minor fit problems, but these do not appear across all reviews.”\n\n"
        f"Here are the reviews:\n{reviews}\n\n"
        "Write the final summary now. Max 50 words."
    )

    return llama_generate(system_prompt, user_prompt, max_new_tokens=120)




best_products["llm_summary"] = best_products.apply(
    summarize_product_row_with_llama,
    axis=1
)

worst_products["llm_summary"] = worst_products.apply(
    summarize_product_row_with_llama,
    axis=1
)

def clean_summary(s: pd.Series) -> pd.Series:
    return (
        s.str.replace('<[^>]+>', '', regex=True)         # remove HTML tags
         .str.replace(r'^[a-zA-Z]\.\s*', '', regex=True) # leading "s." / "t." etc.
         .str.replace(r'\.\s*,', '.', regex=True)        # ". ,"
         .str.replace(r'\n+', ' ', regex=True)           # newlines
         .str.strip()                                    # trim spaces
    )



best_products['llm_summary'] = clean_summary(best_products['llm_summary'])
worst_products['llm_summary'] = clean_summary(worst_products['llm_summary'])


cols = ['predicted_product_category', 'name', 'imageURLs', 'llm_summary']
best_products_with_llm_summary = best_products[cols]
worst_products_with_llm_summary = worst_products[cols]

json_best_data = best_products_with_llm_summary.to_json(orient="records")
json_worst_data = worst_products_with_llm_summary.to_json(orient="records")


with open(Path("data/json_best_data.pkl"), "wb") as f:
    pickle.dump(json_best_data, f)

with open(Path("data/json_worst_data.pkl"), "wb") as f:
    pickle.dump(json_worst_data, f)


print('\n\nModel 3 completed')