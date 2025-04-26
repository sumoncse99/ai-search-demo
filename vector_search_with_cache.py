import pandas as pd
import redis
import hashlib
import json

# Define constants
PRODUCTS_FILE = 'combined_products.parquet'
FIELDS_TO_COMBINE = [
    'category_left', 'brand_left', 'title_left', 'description_left', 'specTableContent_left',
    'brand_right', 'title_right', 'description_right', 'specTableContent_right'
]

# Connect to Redis
cache = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# Load the products
df = pd.read_parquet(PRODUCTS_FILE)

# Combine fields into a searchable text
def preprocess_data():
    df['combined'] = df[FIELDS_TO_COMBINE].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

# Hash the query for Redis
def hash_query(query):
    return hashlib.md5(query.encode()).hexdigest()

# Search in the DataFrame and return clean list
def search_products(query, limit=10):
    key = hash_query(query)
    cached_result = cache.get(key)
    if cached_result:
        print("✅ Returning result from cache...")
        products = json.loads(cached_result)
    else:
        print("🔍 Searching in products...")
        matches = df[df['combined'].str.contains(query, case=False, na=False)]

        products = []
        for _, row in matches.iterrows():
            product = {
                "brand_left": row.get('brand_left', ''),
                "title_left": row.get('title_left', ''),
                "description_left": row.get('description_left', ''),
                "brand_right": row.get('brand_right', ''),
                "title_right": row.get('title_right', ''),
                "description_right": row.get('description_right', ''),
            }
            products.append(product)

        products = products[:limit]

        cache.set(key, json.dumps(products))

    return products

# Prepare the data
preprocess_data()

# Main loop
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
        #print(f"  🔹 Description (Left): {product['description_left']}")
        print(f"  🔹 Brand (Right): {product['brand_right']}")
        print(f"  🔹 Title (Right): {product['title_right']}")
        #print(f"  🔹 Description (Right): {product['description_right']}\n")
