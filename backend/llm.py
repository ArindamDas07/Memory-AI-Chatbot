from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
import os

# ============================================================
# LLM Provider Configuration
# ============================================================
# Using ChatGroq for high-reasoning tasks (Extraction & Conflict Resolution).
# This ensures deterministic structured output even with complex user disclosures.

groq_api_key = os.getenv("GROQ_API_KEY")

# Primary model choice: gpt-oss-120b (via Groq)
# Fallback options: llama-3.3-70b-versatile, llama-3.1-8b-instant
generation_llm = ChatGroq(
    model="openai/gpt-oss-120b", 
    temperature=0
)

# ============================================================
# Memory extraction schema
# ============================================================

class MemoryCandidate(BaseModel):
    """Represent a single candidate personal memory. STRICTLY GROUNDED."""

    should_store: bool = Field(
        description="True ONLY if the message contains a NEW or UPDATED personal fact about the user."
    )

    memory_category: str = Field(
        default="OTHER",
        description="Pick one: IDENTITY, PREFERENCES, CAREER, TRAVEL, GOALS, RELATIONSHIPS, or OTHER."
    )

    memory_key: str = Field(
        default="fact",
        description="A short 1-2 word identifier. If the info fits an existing key you were shown, use that exact key."
    )

    memory_value: str = Field(
        default="",
        description="The actual fact text. STRICT RULE: Only extract what is EXPLICITLY written. Do not guess names or companies."
    )

class MemoryExtraction(BaseModel):
    """Container for multiple facts extracted from a single conversational turn."""
    observations: List[MemoryCandidate] = Field(
        description="A list of all distinct personal facts found in the user's message."
    )

# ============================================================
# Memory comparison schema
# ============================================================

class MemoryComparison(BaseModel):
    """Represent the comparison result between a new fact and existing database records."""

    decision: str = Field(
        description="Must be exactly 'SAME' (no change) or 'UPDATE' (new info replaces old info)."
    )

    reason: Optional[str] = Field(
        default=None,
        description="Brief explanation of the logic behind the decision."
    )

# ============================================================
# Structured Output Wrappers
# ============================================================

# We utilize .with_structured_output to force the LLM into returning Pydantic objects.
memory_extractor_llm = generation_llm.with_structured_output(
    MemoryExtraction
)

memory_comparison_llm = generation_llm.with_structured_output(
    MemoryComparison
)

# ============================================================
# Utility Functions
# ============================================================

def generate_with_llm(prompt: str):
    """Generate a standard natural language response from the core model."""
    response = generation_llm.invoke(prompt)
    return response.content