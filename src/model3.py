import os
import pickle

import pandas as pd
import torch
from dotenv import find_dotenv, load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Globals that will be initialized in main()
tokenizer = None
model = None
device = None


def llama_generate(
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 220,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> str:
    """
    Setup the Llama model
    """

    prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.max_new_tokens = max_new_tokens
    gen_cfg.temperature = temperature
    gen_cfg.top_p = top_p
    gen_cfg.do_sample = False
    gen_cfg.pad_token_id = tokenizer.eos_token_id
    gen_cfg.repetition_penalty = 1.15  # push it away from repeating
    gen_cfg.no_repeat_ngram_size = 6  # avoid repeating long phrases

    with torch.x():
        output_ids = model.generate(
            **inputs,
            generation_config=gen_cfg,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    answer = full_text[len(prompt) :].strip()
    return answer


def summarize_product_row_with_llama(row) -> str:
    """
    Define the model prompts, run the model
    """
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


def clean_summary(s: pd.Series) -> pd.Series:
    """
    Clean up the llama model output
    """
    return (
        s.str.replace("<[^>]+>", "", regex=True)  # remove HTML tags
        .str.replace(r"^[a-zA-Z]\.\s*", "", regex=True)  # leading "s." / "t." etc.
        .str.replace(r"\.\s*,", ".", regex=True)  # ". ,"
        .str.replace(r"\n+", " ", regex=True)  # newlines
        .str.strip()  # trim spaces
    )


def main():
    """
    Main function to run the model
    It loads the pickle files with the best and worst products and generates a Llama product summary
    It returns 2 .json files - best and worst products with the Llama product summary
    """
    print("Model 3 started\n\n")

    _ = load_dotenv(find_dotenv())
    hf_token = os.getenv("HF_TOKEN")

    with open("data/best_products.pkl", "rb") as f:
        best_products = pickle.load(f)

    with open("data/worst_products.pkl", "rb") as f:
        worst_products = pickle.load(f)

    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    global device, tokenizer, model
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    model.eval()

    best_products["llm_summary"] = best_products.apply(
        summarize_product_row_with_llama, axis=1
    )

    worst_products["llm_summary"] = worst_products.apply(
        summarize_product_row_with_llama, axis=1
    )

    best_products["llm_summary"] = clean_summary(best_products["llm_summary"])
    worst_products["llm_summary"] = clean_summary(worst_products["llm_summary"])

    # Select the columns to save
    cols = [
        "predicted_product_category",
        "name",
        "imageURLs",
        "summary_reviews_string",
        "llm_summary",
    ]

    # Save the best and worst products with the Llama product summary
    best_products_with_llm_summary = best_products[cols]
    worst_products_with_llm_summary = worst_products[cols]

    best_products_with_llm_summary.to_json(
        "data/best_products.json", orient="records", indent=2, force_ascii=False
    )

    worst_products_with_llm_summary.to_json(
        "data/worst_products.json", orient="records", indent=2, force_ascii=False
    )

    print("\n\nModel 3 completed")
    return best_products_with_llm_summary, worst_products_with_llm_summary


if __name__ == "__main__":
    main()
