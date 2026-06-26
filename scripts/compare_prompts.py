import os
import logging
logging.disable(logging.CRITICAL)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

while True:
    p1 = input("\nPrompt 1: ")
    p2 = input("Prompt 2: ")

    embeddings = model.encode([p1, p2], show_progress_bar=False)
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    print(f"Cosine Similarity: {sim:.4f}")