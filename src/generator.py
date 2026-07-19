"""
generator.py
------------
The "RAG brain" of the agent. This is the only module that talks to the LLM.

Pipeline for every quiz request:
  1. Retrieve historic facts from ChromaDB (offline, curated knowledge).
  2. Retrieve fresh context from the live web via DuckDuckGo.
  3. Merge both into a single grounding context.
  4. Send a strict, JSON-mode prompt to the LLM so questions are:
       - Grounded only in the retrieved context (no hallucinations)
       - Structured predictably so the UI can render them reliably
       - Fresh/unique on every regeneration request
"""

import json
import random
import time

from src.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_PROVIDER,
    DEFAULT_NUM_QUESTIONS,
)
from src.database import query_historic_facts
from src.search import get_live_news_context


class QuizGenerationError(Exception):
    """Raised when the LLM output cannot be parsed into a valid quiz."""


def _build_unified_context(sport: str) -> str:
    """Gathers and merges ChromaDB + live web context for a given sport."""
    db_query = f"{sport} history cup championships rules records"
    db_matches = query_historic_facts(sport=sport, query_text=db_query, n_results=3)
    db_context = "\n".join(f"- {fact}" for fact in db_matches) if db_matches else (
        "No offline historic data recorded for this sport."
    )

    web_context = get_live_news_context(sport)

    return (
        f"=== HISTORICAL FACTS (ChromaDB) ===\n{db_context}\n\n"
        f"=== LIVE INTERNET NEWS (Web Search) ===\n{web_context}"
    )


def _build_system_prompt(unified_context: str) -> str:
    return (
        "You are an expert sports quiz creator for a social media content team. "
        "You must rely STRICTLY on the CONTEXT provided below. Never invent facts, "
        "scores, dates, or names that are not supported by the context. If the "
        "context is thin, write simpler questions that stay strictly accurate to "
        "what is given rather than guessing.\n\n"
        f"CONTEXT:\n{unified_context}\n\n"
        "Respond with ONLY a valid JSON object (no markdown fences, no preamble, "
        "no commentary) matching exactly this schema:\n"
        "{\n"
        '  "questions": [\n'
        "    {\n"
        '      "question": "string",\n'
        '      "options": {"A": "string", "B": "string", "C": "string", "D": "string"},\n'
        '      "correct_answer": "A" | "B" | "C" | "D",\n'
        '      "explanation": "string, cites the specific context fact used"\n'
        "    }\n"
        "  ]\n"
        "}"
    )


def _build_user_prompt(sport: str, difficulty: str, num_questions: int) -> str:
    # A random seed phrase nudges the model toward generating a fresh set of
    # questions/angles each time, instead of repeating the same quiz.
    variety_seed = random.randint(1000, 9999)
    difficulty_guidance = {
        "Easy": "Focus on well-known, headline facts (winners, founding years, basic rules).",
        "Medium": "Mix well-known facts with slightly more specific details (records, venues, runner-ups).",
        "Hard": "Focus on specific statistics, lesser-known details, and precise figures from the context.",
    }.get(difficulty, "Use a balanced mix of well-known and specific facts.")

    return (
        f"Generate exactly {num_questions} unique multiple-choice questions for the sport: {sport}.\n"
        f"Difficulty target: {difficulty}. {difficulty_guidance}\n"
        f"Variety seed: {variety_seed} (use this only to help vary phrasing/angle, "
        "do not mention it in the output).\n"
        "Every question, option set, and explanation must be answerable strictly "
        "from the CONTEXT given in the system message. Do not repeat the same "
        "question twice. Return ONLY the JSON object described in the schema."
    )


def _call_llm(system_instruction: str, user_prompt: str) -> str:
    """
    Sends the prompt to whichever LLM provider is configured
    (LLM_PROVIDER = "openai" or "gemini") and returns the raw text response.
    Both are asked to return a strict JSON object with no markdown fences.
    """
    if LLM_PROVIDER == "gemini":
        from google import genai
        from google.genai import types # type: ignore

        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.8,
                response_mime_type="application/json",
            ),
        )
        return response.text

    # Default: OpenAI
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _parse_llm_response(raw_text: str) -> list:
    """
    Parses the LLM's JSON response into a Python list of question dicts.
    Strips accidental markdown code fences before parsing, and validates
    that each question has the fields the UI needs.
    """
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).replace("json", "", 1)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise QuizGenerationError(f"Model did not return valid JSON: {e}")

    questions = parsed.get("questions", [])
    if not questions:
        raise QuizGenerationError("Model response contained no questions.")

    validated = []
    for q in questions:
        if not all(k in q for k in ("question", "options", "correct_answer", "explanation")):
            continue
        if not all(letter in q["options"] for letter in ("A", "B", "C", "D")):
            continue
        if q["correct_answer"] not in ("A", "B", "C", "D"):
            continue
        validated.append(q)

    if not validated:
        raise QuizGenerationError("No well-formed questions could be parsed from the model output.")

    return validated


def compile_quiz_data(sport: str, difficulty: str, num_questions: int = DEFAULT_NUM_QUESTIONS,
                       max_retries: int = 2):
    """
    Full RAG pipeline entry point used by the Streamlit app.

    Returns a tuple: (list_of_question_dicts, unified_context_string)
    Raises QuizGenerationError if the LLM output could not be parsed after retries.
    """
    if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
        raise QuizGenerationError(
            "GEMINI_API_KEY is not set. Add it to your .env file (see .env.example)."
        )
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        raise QuizGenerationError(
            "OPENAI_API_KEY is not set. Add it to your .env file (see .env.example)."
        )

    unified_context = _build_unified_context(sport)
    system_instruction = _build_system_prompt(unified_context)

    last_error = None
    for attempt in range(1, max_retries + 2):
        user_prompt = _build_user_prompt(sport, difficulty, num_questions)
        try:
            raw_text = _call_llm(system_instruction, user_prompt)
            questions = _parse_llm_response(raw_text)
            return questions, unified_context
        except QuizGenerationError as e:
            last_error = e
            time.sleep(0.5)  # brief pause before retrying
        except Exception as e:
            last_error = QuizGenerationError(f"LLM API call failed: {e}")
            time.sleep(0.5)

    raise last_error