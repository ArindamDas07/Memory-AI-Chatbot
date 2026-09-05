from typing import TypedDict, List, Optional
from backend.llm import MemoryCandidate

class CompanionState(TypedDict):
    """
    Shared state passed between nodes in the LangGraph workflow.
    Represents the full context of a single conversational turn.
    """

    # --- User & Session Identity ---
    user_id: str
    session_id: str

    # --- Current Interaction ---
    user_message: str
    ai_response: str
    conversation_id: int
    chat_number: int

    # --- Memory Retrieval Context ---
    episodic_memories: List[dict]   # Relevant past chat snippets (Vector search)
    long_term_memories: List[dict] # The "Biography": All active and past facts

    # --- Reasoning Context ---
    context: str
    conversation_embedding: List[float]

    # --- Extraction & Versioning State ---
    # Supports processing multiple facts in one turn
    memory_candidate: List[MemoryCandidate]
    
    # Stores the Judge's decision for each candidate in memory_candidate
    memory_match: Optional[dict]
    memory_decision: Optional[str]
    memory_decision_list: List[dict]