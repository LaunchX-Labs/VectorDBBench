import faiss
import numpy as np
from datasets import load_dataset
import itertools
import json

# Configuration
d = 768                          # Cohere embedding dimension
nlist = 1024                     # Number of clusters
m = 64                           # Sub-vectors (768/64=12D per sub-vector)
nbits = 8                        # Bits per sub-vector
train_size = 100000              # Number of vectors for training
index_id = "cohere_wiki_ivfpq"   # Unique ID for MySQL

# Load Cohere embeddings in streaming mode
streamed_ds = load_dataset(
    "Cohere/wikipedia-22-12-en-embeddings",
    split="train",
    streaming=True
)

# Load training vectors (first 100K)
print(f"Loading {train_size} vectors for training...")
train_batch = list(itertools.islice(streamed_ds, train_size))
train_vectors = np.stack([rec['emb'] for rec in train_batch], axis=0).astype('float32')

print(train_vectors[0])


# Normalize vectors for cosine similarity
faiss.normalize_L2(train_vectors)

quantizer = faiss.IndexFlatIP(d)  # Inner product (cosine similarity)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)

print("Training Started")

# Train on subset (100K normalized vectors)
index.train(train_vectors[:100000])

print("Training completed")

# Extract index data for MySQL
print("Extracting index data...")


# # 1. Get centroids (nlist x d)
# centroids = faiss.vector_to_array(index.quantizer.xb).reshape(-1, d)

# # 2. Get PQ codebooks (m x 256 x (d/m))
# pq = index.pq
# codebooks = faiss.vector_to_array(pq.centroids).reshape(m, 256, -1)
# 1. Get IVF coarse centroids (shape: nlist x d)
# centroids = faiss.vector_to_array(index.quantizer.centroids).reshape(nlist, d)
# Extract centroids correctly
# if hasattr(quantizer, 'xb'):
#     centroids = faiss.vector_to_array(quantizer.xb).reshape(-1, d)
# else:
#     # Alternative method for IndexFlatIP
#     centroids = np.zeros((nlist, d), dtype='float32')
#     quantizer.reconstruct_batch(0, nlist, centroids)
centroids = index.quantizer.reconstruct_n(0, nlist)

# Get the product quantizer object
# pq = index.pq

# Extract codebooks (shape: m x 256 x (d/m))
# codebooks = faiss.vector_to_array(pq.centroids).reshape(pq.m, 256, pq.ds)
codebooks = faiss.vector_to_array(index.pq.centroids).reshape(m, 256, d // m)
# Print shapes for sanity check
print(f"Coarse centroids shape: {centroids.shape}")
print(f"PQ codebooks shape: {codebooks.shape}")

metadata_sql = f"""
INSERT INTO VECTORDB_DATA VALUES (
  '{index_id}', 'metadata', 0,
  JSON_OBJECT('version', 1, 'nlist', {nlist}, 'pq_m', {m}, 'pq_nbits', {nbits})
);
"""

quantizer_sqls = [
    f"INSERT INTO VECTORDB_DATA VALUES ("
    f"'{index_id}', 'quantizer', {i}, '{centroids[i].tolist()}'"
    f");"
    for i in range(nlist)
]

# Product quantizer codebooks
product_quantizer_sqls = []
for m_i in range(m):
    for code in range(256):
        product_quantizer_sqls.append(
            f"INSERT INTO VECTORDB_DATA VALUES ("
            f"'{index_id}', 'product_quantizer', {m_i * 256 + code}, "
            f"'{json.dumps(codebooks[m_i, code].tolist())}'"  # Removed quotes around json.dumps
            f");"
        )


# Combine all SQL
full_sql = (
    metadata_sql + "\n" + 
    "\n".join(quantizer_sqls) + "\n" + 
    "\n".join(product_quantizer_sqls)
)

# Save to file
with open("cohere_wiki_ivfpq.sql", "w") as f:
    f.write(full_sql)

print(f"SQL for auxiliary table saved to cohere_wiki_ivfpq.sql")
print(f"Total centroids: {nlist}")
print(f"Total PQ codebook entries: {m * 256}")