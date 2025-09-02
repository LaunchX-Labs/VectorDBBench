# from datasets import load_dataset

# # Load the English split of Cohere Wikipedia dataset
# dataset = load_dataset("Cohere/wikipedia-22-12", "en")

# # Inspect one record
# print(dataset["train"][0])
# from datasets import load_dataset
# dataset = load_dataset("Cohere/wikipedia-22-12-en-embeddings", split="train")

# embeddings = np.stack([rec['emb'] for rec in dataset], axis=0).astype('float32')
# print(embeddings.shape)  # e.g. (35_000_000, 768)

from datasets import load_dataset
import numpy as np
import itertools

streamed_ds = load_dataset(
    "Cohere/wikipedia-22-12-en-embeddings",
    split="train",
    streaming=True
)

# Example: Take first 50k vectors
batch = list(itertools.islice(streamed_ds, 50000))
embeddings = np.stack([rec['emb'] for rec in batch], axis=0).astype('float32')
print(embeddings.shape)
