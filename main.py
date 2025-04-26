import pandas as pd
import redis
import hashlib
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# Constants
PRODUCTS_FILE = 'combined_products.parquet'
FIELDS_TO_COMBINE = [
    'category_left', 'brand_left', 'title_left', 'specTableContent_left',
    'brand_right', 'title_right', 'specTableContent_right'
]

# Redis Cache
cache = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# Load Data
print("📦 Loading products...")
df = pd.read_parquet(PRODUCTS_FILE)
df['combined'] = df[FIELDS_TO_COMBINE].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

# Embedding Model
print("🔠 Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Vector Embedding + FAISS Index Setup
index_file = "faiss.index"
embedding_file = "embeddings.npy"

if os.path.exists(index_file) and os.path.exists(embedding_file):
    print("📂 Loading existing FAISS index...")
    index = faiss.read_index(index_file)
    embeddings = np.load(embedding_file)
else:
    print("📈 Generating embeddings (chunked)...")

    batch_size = 256
    texts = df['combined'].tolist()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.append(batch_embeddings)

        if (i//batch_size) % 10 == 0:
            print(f"✅ Embedded {i} / {len(texts)}")

    embeddings = np.vstack(all_embeddings)
    np.save(embedding_file, embeddings)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, index_file)
    print("✅ Saved FAISS index and embeddings.")

# Cache Helpers
def hash_query(query):
    return hashlib.md5(query.encode()).hexdigest()

# Search
def search_products(query, limit=10):
    key = hash_query("hybrid:" + query)
    cached_result = cache.get(key)
    if cached_result:
        print("✅ Returning result from cache...")
        return json.loads(cached_result)

    print("🔍 Performing hybrid search...")

    # Keyword Search
    keyword_matches = df[df['combined'].str.contains(query, case=False, na=False)]

    # Vector Search
    query_embedding = model.encode([query], normalize_embeddings=True)
    D, I = index.search(query_embedding, limit)
    vector_matches = df.iloc[I[0]]

    # Combine results (by index) and deduplicate
    hybrid = pd.concat([keyword_matches, vector_matches]).drop_duplicates().head(limit)

    results = []
    for _, row in hybrid.iterrows():
        product = {
            "brand_left": row.get('brand_left', ''),
            "title_left": row.get('title_left', ''),
            "brand_right": row.get('brand_right', ''),
            "title_right": row.get('title_right', ''),
        }
        results.append(product)

    cache.set(key, json.dumps(results))
    return results

# Main loop
if __name__ == "__main__":
    while True:
        query = input("\n🔎 Search query (or type 'exit'): ")
        if query.lower() == 'exit':
            break

        results = search_products(query)

        if not results:
            print("❌ No products found for your search.")
        else:
            total_results = len(results)
            print(f"\n🎯 Showing {min(10, total_results)} of {total_results} product(s):\n")

            for idx, product in enumerate(results[:10], start=1):
                print(f"📦 Product {idx}:")
                print(f"  🔹 Brand (Left): {product['brand_left']}")
                print(f"  🔹 Title (Left): {product['title_left']}")
                print(f"  🔹 Brand (Right): {product['brand_right']}")
                print(f"  🔹 Title (Right): {product['title_right']}\n")
