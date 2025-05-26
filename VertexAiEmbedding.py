from google.cloud import bigquery
from vertexai.preview.language_models import TextEmbeddingModel
import pandas as pd
import numpy as np

# Initialize clients and configs
PROJECT_ID = "charged-sled-459818-s1"
DATASET_ID = "kb_dataset"
SOURCE_TABLE = "ketto_knowledge_base"
DEST_TABLE = "ketto_knowledge_base_with_embeddings"
LOCATION = "US"

bqclient = bigquery.Client(project=PROJECT_ID, location=LOCATION)
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

# Step 1: Query all required fields (assuming they exist in source)
query = f"""
SELECT chunk_id, page_url, page_title, question, answer, chunk_index, content_chunk, token_count
FROM `{PROJECT_ID}.{DATASET_ID}.{SOURCE_TABLE}`
"""
print("Querying data from BigQuery...")
df = bqclient.query(query).to_dataframe()
print(f"Fetched {len(df)} rows.")

# Step 2: Generate embeddings for each content chunk
print("Generating embeddings...")
embeddings = []
for text in df['content_chunk']:
    emb = embedding_model.get_embeddings([text])[0].values
    embeddings.append(emb)

df['embedding'] = embeddings

# Step 3: Convert embeddings to list of floats (required for BigQuery)
df['embedding'] = df['embedding'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

# Step 4: Define full schema including embedding
schema = [
    bigquery.SchemaField("chunk_id", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("page_url", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("page_title", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("question", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("answer", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("chunk_index", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("content_chunk", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("token_count", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
]

table_id = f"{PROJECT_ID}.{DATASET_ID}.{DEST_TABLE}"

job_config = bigquery.LoadJobConfig(
    schema=schema,
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,  # overwrite table every run
)

print(f"Uploading data with embeddings to {table_id}...")
job = bqclient.load_table_from_dataframe(df, table_id, job_config=job_config)
job.result()
print(f"Uploaded {job.output_rows} rows to {table_id}")

print("Done! Your data with embeddings is now in BigQuery.")
