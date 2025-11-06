from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

# Define connection parameters
MILVUS_HOST = os.getenv("MILVUS_HOST",'localhost')#"localhost" # Or "127.0.0.1"
MILVUS_PORT = os.getenv("MILVUS_PORT","19530")#"19530"
CONNECTION_ALIAS = "default" # Default alias

try:
    # Connect to Milvus
    connections.connect(
        alias=CONNECTION_ALIAS,
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )
    print(f"Successfully connected to Milvus instance at {MILVUS_HOST}:{MILVUS_PORT}")

    # Check connection status (optional)
    print(f"Milvus server version: {utility.get_server_version()}")

except Exception as e:
    print(f"Failed to connect to Milvus: {e}")
    # Handle connection error appropriately
    exit()

#--Model Definition --#
field_id=FieldSchema(
    name='chunk_id',
    dtype=DataType.INT64,
    is_primary=True,
    auto_id=True
)

VECTOR_DIMENSION=768

field_vector=FieldSchema(
    name="embedding",
    dtype=DataType.FLOAT_VECTOR,
    dim=VECTOR_DIMENSION
)

# Metadata fields (customize as needed for your RAG data)
field_text_chunk = FieldSchema(
  name="text_chunk",
  dtype=DataType.VARCHAR,
  max_length=65535 # Adjust max length as needed
)

field_source_doc = FieldSchema(
  name="source_document",
  dtype=DataType.VARCHAR,
  max_length=1024
)

# --- Define Schema ---
# Group the fields into a schema
schema = CollectionSchema(
  fields=[field_id, field_vector, field_text_chunk, field_source_doc],
  description="Collection for RAG document chunks",
  enable_dynamic_field=False # Set to True if you want to add fields later without schema changes
)

# --- Define Collection Name ---
COLLECTION_NAME = "rag_documents"

# Check if collection already exists
if utility.has_collection(COLLECTION_NAME, using=CONNECTION_ALIAS):
    print(f"Collection '{COLLECTION_NAME}' already exists.")
    collection = Collection(name=COLLECTION_NAME, using=CONNECTION_ALIAS)
else:
    print(f"Creating collection '{COLLECTION_NAME}'...")
    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema,
        using=CONNECTION_ALIAS,
        consistency_level="Bounded" # Or "Session", "Strong", "Eventually" - Bounded is often a good balance
    )
    print("Collection created successfully.")

# --- Define Index Parameters ---
INDEX_FIELD_NAME = "embedding" # Must match the vector field name in the schema

# Choose an index type and metric type appropriate for your data/use case
# Common choices:
# Index: "HNSW", "IVF_FLAT", "FLAT" (brute-force, only for small datasets)
# Metric: "L2" (Euclidean distance), "IP" (Inner Product / Cosine Similarity if normalized)

index_params = {
  "metric_type": "L2",    # Or "IP"
  "index_type": "HNSW",  # Or "IVF_FLAT"
  "params": {
      "M": 16,              # HNSW parameter (higher means more accuracy, slower build/search)
      "efConstruction": 256 # HNSW parameter (higher means better index quality, slower build)
      # For IVF_FLAT, use params like {"nlist": 1024}
      }
}

# --- Create Index ---
print(f"Creating index for field '{INDEX_FIELD_NAME}'...")
# Check if index already exists first (optional but good practice)
has_index = False
for index in collection.indexes:
    if index.field_name == INDEX_FIELD_NAME:
        print(f"Index on field '{INDEX_FIELD_NAME}' already exists.")
        has_index = True
        break

if not has_index:
    try:
        collection.create_index(
          field_name=INDEX_FIELD_NAME,
          index_params=index_params,
          index_name=f"{INDEX_FIELD_NAME}_idx" # Optional index name
        )
        print("Index created successfully.")
        utility.wait_for_index_building_complete(COLLECTION_NAME, using=CONNECTION_ALIAS)
        print("Index building complete.")
    except Exception as e:
        print(f"Error creating index: {e}")

print(f"Loading collection '{COLLECTION_NAME}' into memory...")
collection.load()
utility.wait_for_loading_complete(COLLECTION_NAME, using=CONNECTION_ALIAS)
print("Collection loaded successfully.")



# --- Example Data Preparation (Replace with your actual data) ---
# Assume you have generated these from your documents:
text_chunks = ["This is the first chunk.", "This is the second piece of text.", "..."]
source_docs = ["doc1.pdf", "doc1.pdf", "..."]
# Assume 'embeddings' is a list of lists (or numpy array) where each inner list is a vector
embeddings = [[0.1, 0.2, ..., 0.9], [0.4, 0.5, ..., 0.3], [...]] # Example, use your real embeddings

# Structure data for insertion based on schema (excluding auto_id primary key)
# Ensure the order matches within lists: embedding[i] belongs to text_chunks[i]
data_to_insert = [
    embeddings,       # Corresponds to field_vector ("embedding")
    text_chunks,      # Corresponds to field_text_chunk ("text_chunk")
    source_docs       # Corresponds to field_source_doc ("source_document")
]

# --- Insert Data ---
print(f"Inserting {len(text_chunks)} entities...")
insert_result = collection.insert(data_to_insert)
print(f"Insertion result: {insert_result}")

# Optional: Flush data to make it searchable immediately if needed
# collection.flush()
# print("Data flushed.")
# Note: Milvus automatically flushes periodically. Explicit flush ensures immediate visibility.