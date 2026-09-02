from typing import Optional

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama


# ============================================================
# Main generation LLM
# ============================================================

generation_llm = ChatOllama(
    model="gemma3:4b",
    temperature=0
)


# ============================================================
# Memory extraction schema
# ============================================================

class MemoryCandidate(BaseModel):
    """Represent a candidate personal memory extracted from a user message."""

    should_store: bool = Field(
        description="Whether the user's message contains a persistent personal fact worth storing."
    )

    memory_key: Optional[str] = Field(
        default=None,
        description="Short category/key for the personal memory."
    )

    memory_value: Optional[str] = Field(
        default=None,
        description="Concise description of the user's personal fact."
    )

    reason: Optional[str] = Field(
        default=None,
        description="Brief explanation for the decision."
    )


# ============================================================
# Memory comparison schema
# ============================================================

class MemoryComparison(BaseModel):
    """Represent the comparison result between two long-term memories."""

    decision: str = Field(
        description="Must be exactly SAME or UPDATE."
    )

    reason: Optional[str] = Field(
        default=None,
        description="Brief explanation of the decision."
    )


# ============================================================
# Structured memory extractor
# ============================================================

memory_extractor_llm = generation_llm.with_structured_output(
    MemoryCandidate
)


# ============================================================
# Structured memory comparison
# ============================================================

memory_comparison_llm = generation_llm.with_structured_output(
    MemoryComparison
)


# ============================================================
# Normal response generation
# ============================================================

def generate_with_llm(prompt: str):
    """Generate a normal conversational response from the LLM."""

    response = generation_llm.invoke(prompt)

    return response.content