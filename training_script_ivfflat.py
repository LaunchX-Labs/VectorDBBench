import faiss
import numpy as np
from datasets import load_dataset
import itertools
import json
import pickle

d = 768                          # Cohere embedding dimension
nlist = 256                    # Number of clusters

train_size = 298000             # Number of vectors for training
index_id = "cohere_wiki_ivfflat"   # Unique ID for MySQL

print("=== IVFFlat Index Configuration ===")
print(f"Dimension (d): {d}")
print(f"Number of clusters (nlist): {nlist}")
print(f"Training size: {train_size}")
print(f"Index ID: {index_id}")

# # Load Cohere embeddings in streaming mode
# streamed_ds = load_dataset(
#     "Cohere/wikipedia-22-12-en-embeddings",
#     split="train",
#     streaming=True
# )

# # Load training vectors (first 100K)
# print(f"Loading {train_size} vectors for training...")
# train_batch = list(itertools.islice(streamed_ds, train_size))
# train_vectors = np.stack([rec['emb'] for rec in train_batch], axis=0).astype('float32')
try:
    with open("accumulated_cohere_embeddings.pkl", 'rb') as f:
        embedding_data = pickle.load(f)
    
    # Convert to numpy array (same format as original Cohere loading)
    train_vectors = np.stack(embedding_data['embeddings'], axis=0).astype('float32')
    
    print(f"Loaded {len(train_vectors)} training vectors")
    print(f"Training vectors shape: {train_vectors.shape}")
    print(f"First vector sample: {train_vectors[0][:10]}...")  # Show first 10 dimensions

except Exception as e:
    print(f"Error loading training vectors from file: {e}")

# Normalize vectors for cosine similarity
faiss.normalize_L2(train_vectors)
print("Vectors normalized for cosine similarity")

quantizer = faiss.IndexFlatIP(d)                   # L2 distance metric
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

print("Training Started")

# Train on subset (100K normalized vectors)
index.train(train_vectors)

# Extract index data for MySQL
print("Extracting index data...")


centroids = index.quantizer.reconstruct_n(0, nlist)

print(f"Coarse centroids shape: {centroids.shape}")

# Sanity check
assert centroids.shape == (nlist, d), f"Centroids shape mismatch: {centroids.shape}"
print("✓ Shape verification passed!")

# Verify centroids are normalized (for cosine similarity)
centroid_norms = np.linalg.norm(centroids, axis=1)
print(f"Centroid norms - min: {centroid_norms.min():.6f}, max: {centroid_norms.max():.6f}, mean: {centroid_norms.mean():.6f}")

metadata_sql = f"""
INSERT INTO VECTORDB_DATA VALUES (
  '{index_id}', 'metadata', 0,
  JSON_OBJECT('version', 1, 'nlist', {nlist})
);
"""

quantizer_sqls = [
    f"INSERT INTO VECTORDB_DATA VALUES ("
    f"'{index_id}', 'quantizer', {i}, '{centroids[i].tolist()}'"
    f");"
    for i in range(nlist)
]

# Combine all SQL
full_sql = (
    metadata_sql + "\n" + 
    "\n".join(quantizer_sqls)
)

# Save to file
with open("cohere_wiki_ivfflat_cosine.sql", "w") as f:
    f.write(full_sql)

print(f"SQL for auxiliary table saved to cohere_wiki_ivfflat_cosine.sql")
print(f"Total centroids: {nlist}")