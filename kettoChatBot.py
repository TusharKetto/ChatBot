# KettoChatbot.py
from google.cloud import bigquery
from vertexai.preview.language_models import TextEmbeddingModel
from google.oauth2 import service_account
from google import genai
from google.genai import types

import textwrap
import datetime
import warnings
import logging

# Suppress warnings and logs
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)

# Constants
PROJECT_ID = "charged-sled-459818-s1"
LOCATION = "us-central1"
BIGQUERY_LOCATION = "US"
DATASET_ID = "kb_dataset"
TABLE_NAME = "ketto_knowledge_base_with_embeddings"
EMBEDDING_MODEL_NAME = "text-embedding-005"
TOP_K = 5
MAX_CONTEXT_CHARS = 4000

class KettoChatbot:
    def __init__(self):
        credentials = service_account.Credentials.from_service_account_file(
            "charged-sled-459818-s1-12e648f06659.json"
        )
        self.bq_client = bigquery.Client(project=PROJECT_ID, location=BIGQUERY_LOCATION, credentials=credentials)
        self.embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
        self.genai_client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")
        self.model = "models/gemini-1.5-pro-preview-0409"  # More stable version

    def embed_query(self, query_text: str):
        embedding = self.embedding_model.get_embeddings([query_text])[0].values
        return embedding if isinstance(embedding, list) else embedding.tolist()

    def search_similar_chunks(self, query_embedding):
        embedding_str = str(query_embedding)
        sql = f"""
        SELECT
            page_url,
            page_title,
            content_chunk,
            ML.DISTANCE(embedding, {embedding_str}, 'COSINE') AS similarity
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}`
        ORDER BY similarity ASC
        LIMIT {TOP_K}
        """
        return self.bq_client.query(sql).to_dataframe()

    def log_response(self, user_query, response_text):
        timestamp = datetime.datetime.now().isoformat()
        with open("llm_responses.log", "a") as f:
            f.write(f"[{timestamp}]\nQuery: {user_query}\nResponse: {response_text}\n\n")

    def query_llm(self, user_query: str) -> str:
        try:
            query_embedding = self.embed_query(user_query)
            relevant_chunks = self.search_similar_chunks(query_embedding)

            if relevant_chunks.empty:
                return "Sorry, I couldn't find relevant information in the knowledge base."

            context = "\n\n---\n\n".join(
                f"{row.content_chunk}" for _, row in relevant_chunks.iterrows()
            )
            if len(context) > MAX_CONTEXT_CHARS:
                context = context[:MAX_CONTEXT_CHARS]

            few_shot_example = """
Context:
[Withdrawals]
Funds raised on behalf of an NGO cannot be transferred to a personal or corporate account. If there is an issue found with the NGO's documents, a refund will be made to the donors.

User Question:
How do NGOs withdraw funds from Ketto?

Answer:
NGOs must withdraw funds directly to their official NGO bank accounts. If Ketto finds issues with the documentation, the funds are refunded to the donors rather than being disbursed.
"""

            prompt = f"""
You are a friendly and knowledgeable assistant helping users understand how Ketto works.

Below is some information from Ketto’s official knowledge base. Based on this, answer the user's question in a clear, concise, and conversational tone. You may summarize or combine relevant points. Do not copy chunks directly unless needed.

{few_shot_example}

Context:
{context}

User Question:
{user_query}

Your Answer:
""".strip()

            contents = [
                types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
            ]

            config = types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=2048,
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                ]
            )

            response = self.genai_client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )

            if not response or not response.candidates:
                raise ValueError("No response candidates received from Gemini.")

            final_answer = response.candidates[0].content.parts[0].text.strip()
            self.log_response(user_query, final_answer)
            return final_answer

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error("Error in query_llm", exc_info=True)
            return f"Error: {str(e)}"
