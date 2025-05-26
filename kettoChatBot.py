from google.cloud import bigquery
from vertexai.preview.language_models import TextEmbeddingModel
from google.oauth2 import service_account
from google import genai
from google.genai import types

import textwrap
import datetime
import warnings
import logging
from io import StringIO
import sys

# Suppress warnings and logs
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)

# === Configuration ===
PROJECT_ID = "charged-sled-459818-s1"
BIGQUERY_LOCATION = "US"
DATASET_ID = "kb_dataset"
TABLE_NAME = "ketto_knowledge_base_with_embeddings"
EMBEDDING_MODEL_NAME = "text-embedding-005"
TOP_K = 5
SERVICE_ACCOUNT_PATH = "/Users/admin/Documents/DataPipleline/charged-sled-459818-s1-6f70737be853.json"

class KettoChatbot:
    def __init__(self):
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_PATH)
        self.bq_client = bigquery.Client(project=PROJECT_ID, location=BIGQUERY_LOCATION, credentials=credentials)
        self.embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
        self.genai_client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")
        self.model = "gemini-2.5-pro-preview-05-06"

    def embed_query(self, query_text: str):
        embedding = self.embedding_model.get_embeddings([query_text])[0].values
        return embedding if isinstance(embedding, list) else embedding.tolist()

    def search_similar_chunks(self, query_embedding):
        embedding_str = str(query_embedding)
        sql = f"""
        SELECT
            page_url,
            page_title,
            question,
            content_chunk,
            ML.DISTANCE(embedding, {embedding_str}, 'COSINE') AS similarity
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}`
        ORDER BY similarity ASC
        LIMIT {TOP_K}
        """
        return self.bq_client.query(sql).to_dataframe()

    def build_prompt(self, user_query, context_df):
        context = "\n\n---\n\n".join(
            f"[{row.question}]\n{row.content_chunk}" for _, row in context_df.iterrows()
        )
        prompt = f"""
YYou are an expert assistant for Ketto, trained to help users with donations, fundraising, verification, and platform usage.

Your task is to synthesize an answer using the context provided below.

Be sure to:
- Focus on the user's question
- Summarize and personalize the response using your own words
- Keep it clear, accurate, and friendly
- Do **not** copy-paste from the context — instead, rephrase for clarity

Only use the context if it is relevant.

Context:
{context}

Question:
{user_query}

Answer in a clear, concise, and helpful manner.
        """.strip()
        return prompt

    def stream_answer(self, prompt):
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        config = types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=4096,
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            ]
        )
        for chunk in self.genai_client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                print(chunk.text.strip(), end=" ", flush=True)

    def generate_answer_text(self, prompt):
        # Captures streamed output and returns it as a string
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        config = types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=4096,
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            ]
        )

        full_response = ""
        for chunk in self.genai_client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                full_response += chunk.text.strip() + " "
        return full_response.strip()

    def log_response(self, user_query, response_text):
        timestamp = datetime.datetime.now().isoformat()
        with open("llm_responses.log", "a") as f:
            f.write(f"[{timestamp}]\nQuery: {user_query}\nResponse: {response_text}\n\n")

    def chat(self, user_query: str):
        try:
            query_embedding = self.embed_query(user_query)
            relevant_chunks = self.search_similar_chunks(query_embedding)
            if relevant_chunks.empty:
                print("Sorry, I couldn't find relevant information in the knowledge base.")
                self.log_response(user_query, "No relevant info found.")
                return
            prompt = self.build_prompt(user_query, relevant_chunks)
            print("\nChatbot answer:\n")
            self.stream_answer(prompt)
            self.log_response(user_query, "[streamed answer]")
        except Exception as e:
            print("Sorry, something went wrong while processing your request.")
            self.log_response(user_query, f"ERROR: {str(e)}")

    def query_llm(self, user_query: str):
        # Used by LLMQuery.py or other integrations
        try:
            query_embedding = self.embed_query(user_query)
            relevant_chunks = self.search_similar_chunks(query_embedding)
            if relevant_chunks.empty:
                return "Sorry, I couldn't find relevant information in the knowledge base."
            prompt = self.build_prompt(user_query, relevant_chunks)
            response = self.generate_answer_text(prompt)
            self.log_response(user_query, response)
            return response
        except Exception as e:
            error_msg = f"Sorry, an error occurred: {e}"
            self.log_response(user_query, error_msg)
            return error_msg
