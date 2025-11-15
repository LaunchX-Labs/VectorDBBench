import faiss
import numpy as np
from datasets import load_dataset
import itertools
import json
import pickle

# Configuration
d = 768                          # Cohere embedding dimension
nlist = 1024                    # Number of clusters (increased for better granularity)
m = 64                           # Sub-vectors (768/64=12D per sub-vector)
nbits = 8                        # Bits per sub-vector (256 centroids per subquantizer)
train_size = 298000             # Increased training size (recommended: 30-100x nlist)
index_id = "cohere_wiki_ivfpq"   # Updated ID to reflect 4K clusters

print("=== IVFPQ Index Configuration (Inner Product) ===") 
print(f"Dimension (d): {d}")
print(f"Number of clusters (nlist): {nlist}")
print(f"Sub-vectors (m): {m}")
print(f"Sub-vector dimension (d/m): {d//m}")
print(f"Bits per sub-vector (nbits): {nbits}")
print(f"Training size: {train_size}")
print(f"Index ID: {index_id}")
print("Similarity metric: Inner Product (no normalization)")

# Verify configuration
if d % m != 0:
    raise ValueError(f"Dimension {d} must be divisible by number of sub-vectors {m}")

# Load training vectors from pickle file
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

# NOTE: NO NORMALIZATION for pure inner product similarity
# The vectors remain in their original space for inner product computation
print("Using raw vectors for inner product similarity (no normalization)")

# Create IVFPQ index for inner product
print("\n=== Creating IVFPQ Index (Inner Product) ===")
quantizer = faiss.IndexFlatIP(d)  # Inner product quantizer
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)

print("Index created. Starting training...")

# Train the index
index.train(train_vectors)
print("Training completed successfully!")

# Verify index parameters
print(f"\n=== Index Parameters Verification ===")
print(f"Index nlist: {index.nlist}")
print(f"Index dimension: {index.d}")
print(f"Index metric: Inner Product")
print(f"PQ M (subquantizers): {index.pq.M}")
print(f"PQ dsub (subvector dim): {index.pq.dsub}")
print(f"PQ nbits: {index.pq.nbits}")
print(f"PQ ksub (centroids per subquantizer): {1 << index.pq.nbits}")

# Extract index data for MySQL
print("\n=== Extracting Index Data ===")

# 1. Extract IVF coarse centroids (shape: nlist x d)
print("Extracting coarse centroids...")
centroids = index.quantizer.reconstruct_n(0, nlist)

# 2. Extract PQ codebooks (shape: m x ksub x d_sub)
print("Extracting PQ codebooks...")
ksub = 1 << index.pq.nbits  # 256 for 8-bit
d_sub = index.pq.dsub       # subvector dimension
codebooks = faiss.vector_to_array(index.pq.centroids).reshape(index.pq.M, ksub, d_sub)

# Verify extraction
print(f"Coarse centroids shape: {centroids.shape} (expected: {nlist} x {d})")
print(f"PQ codebooks shape: {codebooks.shape} (expected: {m} x {ksub} x {d_sub})")

# Sanity checks
assert centroids.shape == (nlist, d), f"Centroids shape mismatch: {centroids.shape}"
assert codebooks.shape == (m, ksub, d_sub), f"Codebooks shape mismatch: {codebooks.shape}"
assert d_sub * m == d, f"Subvector dimensions don't match: {d_sub} * {m} = {d_sub * m} != {d}"

print("✓ All shape verifications passed!")

metadata_sql = f"""
INSERT INTO VECTORDB_DATA VALUES (
  '{index_id}', 'metadata', 0,
  JSON_OBJECT('version', 1, 'nlist', {nlist}, 'pq_m', {m}, 'pq_nbits', {nbits})
);
"""

quantizer_sqls = [
    f"INSERT INTO VECTORDB_DATA VALUES ("
    f"'{index_id}', 'quantizer', {i}, '{json.dumps(centroids[i].tolist())}'"
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
with open("cohere_wiki_ivfpq_ip.sql", "w") as f:
    f.write(full_sql)

print(f"SQL for auxiliary table saved to cohere_wiki_ivfpq_ip.sql")
print(f"Total centroids: {nlist}")
print(f"Total PQ codebook entries: {m * 256}")
