from google.cloud import bigquery
from vertexai.preview.language_models import TextEmbeddingModel
from google.oauth2 import service_account
from google import genai
from google.genai import types
from kettoChatBot import KettoChatbot


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

chatbot_instance = KettoChatbot()
def query_llm(user_query: str) -> str:
    """
    Wrapper function to query the KettoChatbot instance.
    Used by the FastAPI app to return LLM responses.
    """
    return chatbot_instance.query_llm(user_query)