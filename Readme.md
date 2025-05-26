project:
  title: "Ketto Knowledge Base Scraper & Vector Embedding Pipeline"
  description: >
    This project scrapes the Ketto knowledge base, processes the content into chunks,
    and prepares it for ingestion into Google BigQuery and Vertex AI for embedding and semantic search.

setup:
  steps:
    - step: "Install Python 3"
      commands:
        - "brew install python"
        - "python3 --version  # Should output Python 3.13.3"
    - step: "Unzip Project Files"
      commands:
        - "unzip project.zip"
        - "cd project-folder"
    - step: "Create and Activate Virtual Environment"
      commands:
        - "python3 -m venv myenv"
        - "source myenv/bin/activate"
    - step: "Install Required Packages"
      commands:
        - "pip install -r requirements.txt"

pipeline:
  - stage: "Scrape Data"
    description: "Extract all articles from the Ketto knowledge base."
    file_to_run: "ScrapeWeb.py"
    output: "ketto_full_kb_recursive.csv"
    command: "python ScrapeWeb.py"

  - stage: "Split Data into Chunks"
    description: "Split long articles into smaller, manageable chunks for embedding."
    file_to_run: "ChunckSplitting.py"
    output: "ketto_kb_chunks.csv"
    command: "python ChunckSplitting.py"

  - stage: "Ingest to BigQuery & Trigger Embedding"
    description: >
      Push chunked CSV data to BigQuery and trigger the Vertex AI embedding model
      named 'VectorAiEmbedding'.
    file_to_run: "VertexAiEmbedding.py"
    outputs:
      - "Data in BigQuery"
      - "Embeddings created by Vertex AI"
    command: "python VertexAiEmbedding.py"

  - stage: "Generate User Query Embeddings"
    description: "Run this to create vector representations for user queries."
    file_to_run: "UserQueryEmbeddings.py"
    output: "user_query_embedding.npy"
    command: "python UserQueryEmbeddings.py"

  - stage: "Run Chatbot API Server"
    description: "Launch FastAPI server to interact with the chatbot."
    file_to_run: "main.py"
    command: "uvicorn main:app --reload"

  

gcp_service_account_setup:
  instructions:
    - "Go to: https://console.cloud.google.com/iam-admin/serviceaccounts"
    - "Select your project"
    - "Click 'Create Service Account' or use an existing one"
    - "Assign roles: BigQuery Data Viewer, Vertex AI User, etc."
    - "Navigate to 'Keys' tab > 'Add Key' > Create new key (JSON)"
    - "Download the JSON key file"
  set_env_variable:
    description: "Set this environment variable to authenticate GCP services"
    command: export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-file.json"
    example: export GOOGLE_APPLICATION_CREDENTIALS="/Users/admin/Downloads/my-ketto-key.json"

output_summary:
  - step: "Scraping"
    file: "ketto_full_kb_recursive.csv"
  - step: "Chunk Splitting"
    file: "ketto_kb_chunks.csv"
  - step: "BQ Upload & Embedding"
    result: "BigQuery Table + Vertex AI Embeddings"
  - step: "User Embeddings"
    result: "Embeddings for user queries"

project_structure:
  files:
    - "ScrapeWeb.py"
    - "ChunckSplitting.py"
    - "VertexAiEmbedding.py"
    - "UserQueryEmbeddings.py"
    - "LLMQuery.py"
    - "requirements.txt"
    - "README.md"


