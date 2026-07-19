"""
config.py
---------
Centralized configuration loader. Reads environment variables from a local
.env file (never committed to git) so API keys stay out of source code.
"""

import os
from dotenv import load_dotenv

# Load variables from a .env file in the project root, if present.
load_dotenv()

# --- LLM Provider Settings ---
# LLM_PROVIDER can be "openai" or "gemini". Gemini has a free tier that
# doesn't require a billing card, so it's a good default for local testing.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Auto-detect provider: prefer Gemini if its key is set, otherwise OpenAI.
LLM_PROVIDER = os.getenv("LLM_PROVIDER") or ("gemini" if GEMINI_API_KEY else "openai")

# --- Vector DB Settings ---
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "sports_history")

# --- Data Settings ---
SPORTS_FACTS_PATH = os.getenv("SPORTS_FACTS_PATH", "./data/sports_facts.json")

# --- Quiz Generation Settings ---
DEFAULT_NUM_QUESTIONS = int(os.getenv("DEFAULT_NUM_QUESTIONS", "5"))
SUPPORTED_SPORTS = ["Cricket", "Football", "Tennis", "Badminton", "Basketball"]
SUPPORTED_DIFFICULTIES = ["Easy", "Medium", "Hard"]


def validate_config():
    """
    Quick sanity check run at app startup so missing configuration fails
    loudly and early instead of causing a confusing error mid-request.
    """
    warnings = []
    if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
        warnings.append(
            "GEMINI_API_KEY is missing. Copy .env.example to .env and add your free "
            "Gemini key from https://aistudio.google.com/app/apikey."
        )
    elif LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        warnings.append(
            "OPENAI_API_KEY is missing. Copy .env.example to .env and add your key."
        )
    return warnings