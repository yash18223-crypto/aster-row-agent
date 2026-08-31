"""Settings for Aster & Row AI Support Agent."""

import os
from dotenv import load_dotenv

load_dotenv()

# AI model settings
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

MODEL = "openai/gpt-oss-120b"   # LLM to use
MAX_TOKENS = 2048

# Knowledge base search settings
KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge-base")
TOP_K_CHUNKS = 5
SIMILARITY_THRESHOLD = 0.05   # Minimum score to include a document

# Order database
ORDERS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "orders.json")

# Don't show these fields to customers
FORBIDDEN_ORDER_FIELDS = {"email", "shipping_address", "risk_score",
                          "warehouse_note", "support_tags", "internal"}

# How much to trust different document types
AUTHORITY_WEIGHTS = {
    "official": 1.0,
    "none": 0.0,
}

STATUS_WEIGHTS = {
    "active": 1.0,
    "superseded": 0.0,   # Old documents - don't use
    "draft": 0.0,         # Work in progress - don't use
}

AUDIENCE_WEIGHTS = {
    "customer": 1.0,
    "internal": 0.0,      # Hidden from customers
}

# Logging
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
