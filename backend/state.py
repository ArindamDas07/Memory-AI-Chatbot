from typing import TypedDict, List, Optional

from backend.llm import MemoryCandidate


class CompanionState(TypedDict):
    """Shared state passed between nodes in the conversation graph."""

    # --------------------------------------------------------
    # User / Session
    # --------------------------------------------------------

    user_id: str
    session_id: str

    # --------------------------------------------------------
    # Current conversation
    # --------------------------------------------------------

    user_message: str
    ai_response: str

    conversation_id: int
    chat_number: int

    # --------------------------------------------------------
    # Retrieved memories
    # --------------------------------------------------------

    episodic_memories: List[dict]
    long_term_memories: List[dict]

    # --------------------------------------------------------
    # Context for LLM
    # --------------------------------------------------------

    context: str

    # --------------------------------------------------------
    # Conversation embedding
    # --------------------------------------------------------

    conversation_embedding: List[float]

    # --------------------------------------------------------
    # Long-term memory extraction
    # --------------------------------------------------------

    memory_candidate: Optional[MemoryCandidate]

    # --------------------------------------------------------
    # Memory comparison
    # --------------------------------------------------------

    memory_match: Optional[dict]
    memory_decision: Optional[str]