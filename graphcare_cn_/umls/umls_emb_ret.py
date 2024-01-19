import os
import time
import pickle
from tqdm import tqdm
from ChatGPT_emb import embedding_retriever
from concurrent.futures import ThreadPoolExecutor

SAVE_INTERVAL = 5000  # Save after processing every SAVE_INTERVAL names
MAX_RETRIES = 100  # Retry up to MAX_RETRIES times if there's an error

if not os.path.exists("../../exp_data/"):
    os.makedirs("../../exp_data/")

# Load previous embeddings if they exist
try:
    with open('../../exp_data/umls_ent_emb_.pkl', 'rb') as f:
        umls_ent_emb = pickle.load(f)
except FileNotFoundError:
    umls_ent_emb = []

# Loading and preprocessing the names
with open("../../KG_mapping/umls/concept_names.txt", 'r') as f:
    umls_ent = f.readlines()

umls_names = [line.split('\t')[1][:-1] for line in umls_ent]

# Skip names that are already processed
umls_names = umls_names[len(umls_ent_emb):]

def get_embedding(name):
    for _ in range(MAX_RETRIES):
        try:
            emb = embedding_retriever(term=name)
            return emb
        # except KeyError:
        #   pass # Retry on KeyError
        except Exception:
            time.sleep(1)  # Retry on Any Error
    return "Error: Failed to retrieve embedding for {}".format(name)

# Use ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:
    for idx, emb in enumerate(tqdm(executor.map(get_embedding, umls_names), total=len(umls_names))):
        umls_ent_emb.append(emb)
        
        # Periodically save the data
        if (idx + 1) % SAVE_INTERVAL == 0:
            with open('../../exp_data/umls_ent_emb_.pkl', 'wb') as f:
                pickle.dump(umls_ent_emb, f)

# Save the final data
with open('../../exp_data/umls_ent_emb_.pkl', 'wb') as f:
    pickle.dump(umls_ent_emb, f)
