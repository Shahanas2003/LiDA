from dotenv import load_dotenv
import os
load_dotenv()
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

import pandas as pd
import os
import google.generativeai as genai


print(list(genai.list_models()))



# Load API key securely
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("models/gemini-2.5-flash")

def paraphrase_with_gemini(text, num_return=2):
    prompt = f"Give me {num_return} different paraphrases of this line, one per line: \"{text}\""
    response = model.generate_content(prompt)
    return [l.strip() for l in response.text.split('\n') if l.strip()]

df = pd.read_csv("sample_for_gemini.csv")  # A small CSV of sentences & labels

augmented = {"sentence": [], "label": []}
for idx, row in df.iterrows():
    orig, label = row["sentence"], row["label"]
    try:
        paraphrases = paraphrase_with_gemini(orig, num_return=2)
        for para in paraphrases:
            augmented["sentence"].append(para)
            augmented["label"].append(label)
    except Exception as e:
        print(f"Error for row {idx}: {e}")

aug_df = pd.DataFrame(augmented)
aug_df.to_csv("gemini_augmented.csv", index=False)
print("✅ Done: Paraphrases written to gemini_augmented.csv")
