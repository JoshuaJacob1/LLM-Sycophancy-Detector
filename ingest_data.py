import os
import json
import urllib.request
import ssl
import pandas as pd

ssl._create_default_https_context = ssl._create_unverified_context


DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# We will use the Anthropic model-written-evals sycophancy dataset
# specifically the NLP survey dataset which asks technical NLP questions where the user has a stated preference
DATASET_URL = "https://raw.githubusercontent.com/anthropics/evals/main/sycophancy/sycophancy_on_nlp_survey.jsonl"
LOCAL_FILE = os.path.join(DATA_DIR, "sycophancy_nlp_survey.jsonl")

def download_data():
    if not os.path.exists(LOCAL_FILE):
        print(f"Downloading dataset from {DATASET_URL}...")
        urllib.request.urlretrieve(DATASET_URL, LOCAL_FILE)
        print("Download complete.")
    else:
        print("Dataset already exists locally.")

def load_and_parse():
    data = []
    with open(LOCAL_FILE, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    df = pd.DataFrame(data)
    
    # The dataset has 'question', 'answer_matching_behavior' (the sycophantic answer), 
    # and 'answer_not_matching_behavior' (the honest answer)
    print(f"Loaded {len(df)} records.")
    print("Sample record:")
    print(df.iloc[0]['question'])
    print("Sycophantic Answer:", df.iloc[0]['answer_matching_behavior'])
    print("Honest Answer:", df.iloc[0]['answer_not_matching_behavior'])
    
    return df

if __name__ == "__main__":
    download_data()
    df = load_and_parse()
    
    # Save a clean CSV version for easier viewing
    csv_path = os.path.join(DATA_DIR, "sycophancy_nlp_survey_clean.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved parsed data to {csv_path}")
