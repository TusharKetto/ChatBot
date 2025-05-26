from vertexai.preview.language_models import TextEmbeddingModel
from google.cloud import bigquery

PROJECT_ID = "charged-sled-459818-s1"
DATASET_ID = "kb_dataset"
TABLE = "ketto_knowledge_base_with_embeddings"
bqclient = bigquery.Client(project=PROJECT_ID)

embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

query_text = "How can NGOs withdraw funds from Ketto?"
query_embedding = embedding_model.get_embeddings([query_text])[0].values
query_embedding_str = "ARRAY[" + ",".join(map(str, query_embedding)) + "]"

sql = f"""
SELECT
  chunk_id,
  page_url,
  page_title,
  question,
  answer,
  chunk_index,
  content_chunk,
  token_count,
  ML.DISTANCE(embedding, {query_embedding_str}, 'COSINE') AS similarity
FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE}`
ORDER BY similarity ASC
LIMIT 5
"""

print("Running similarity query...")
results_df = bqclient.query(sql).to_dataframe()

for i, row in results_df.iterrows():
    print(f"\nResult {i+1}")
    print("Chunk ID:", row["chunk_id"])
    print("Title:", row["page_title"])
    print("Question:", row["question"])
    print("Answer:", row["answer"])
    print("Chunk Index:", row["chunk_index"])
    print("Token Count:", row["token_count"])
    print("Similarity Score:", row["similarity"])
    print("Content:\n", row["content_chunk"])
