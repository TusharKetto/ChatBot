from google.cloud import bigquery
import pandas as pd

client = bigquery.Client(project="charged-sled-459818-s1")

# Load CSV
df = pd.read_csv("ketto_kb_chunks.csv")

# Define table ID
table_id = "charged-sled-459818-s1.kb_dataset.ketto_knowledge_base"

# Configure job to overwrite the table and update schema
job_config = bigquery.LoadJobConfig(
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
)

# Upload to BigQuery, replacing the existing table
job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
job.result()

print("✅ Upload complete - table replaced with new data and schema")
