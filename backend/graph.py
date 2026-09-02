from typing import Literal

from langgraph.graph import StateGraph, START, END

from backend.state import CompanionState

from backend.nodes import (
retrieve_episodic_memory,
retrieve_long_term_memory,
generate_response,
save_conversation,
generate_conversation_embedding,
store_conversation_embedding,
extract_memory,
compare_candidate_memory,
save_long_term_memory,
)

# ============================================================

# Memory Decision Router

# ============================================================
"""Route memory processing based on the comparison decision."""
def route_memory_decision(
    state: CompanionState,
    ) -> Literal["save_long_term_memory", "end"]:
      

    decision = state.get("memory_decision")

    print("\nMemory decision router:")
    print("Decision:", decision)

    if decision in ("NEW", "UPDATE"):
        return "save_long_term_memory"

    if decision in ("SAME", None):
        return "end"

    raise ValueError(
        f"Invalid memory decision: {decision}"
    )


# ============================================================

# Build Graph

# ============================================================

builder = StateGraph(CompanionState)

# ============================================================

# Nodes

# ============================================================

builder.add_node(
"retrieve_episodic_memory",
retrieve_episodic_memory
)

builder.add_node(
"retrieve_long_term_memory",
retrieve_long_term_memory
)

builder.add_node(
"generate_response",
generate_response
)

builder.add_node(
"save_conversation",
save_conversation
)

builder.add_node(
"generate_conversation_embedding",
generate_conversation_embedding
)

builder.add_node(
"store_conversation_embedding",
store_conversation_embedding
)

builder.add_node(
"extract_memory",
extract_memory
)

builder.add_node(
"compare_candidate_memory",
compare_candidate_memory
)

builder.add_node(
"save_long_term_memory",
save_long_term_memory
)

# ============================================================

# Main Conversation Flow

# ============================================================

builder.add_edge(
START,
"retrieve_episodic_memory"
)

builder.add_edge(
"retrieve_episodic_memory",
"retrieve_long_term_memory"
)

builder.add_edge(
"retrieve_long_term_memory",
"generate_response"
)

builder.add_edge(
"generate_response",
"save_conversation"
)

builder.add_edge(
"save_conversation",
"generate_conversation_embedding"
)

builder.add_edge(
"generate_conversation_embedding",
"store_conversation_embedding"
)

builder.add_edge(
"store_conversation_embedding",
"extract_memory"
)

builder.add_edge(
"extract_memory",
"compare_candidate_memory"
)

# ============================================================

# Conditional Memory Flow

# ============================================================

builder.add_conditional_edges(
"compare_candidate_memory",
route_memory_decision,
{
"save_long_term_memory": "save_long_term_memory",
"end": END,
}
)

# ============================================================

# Finish Long-Term Memory Operation

# ============================================================

builder.add_edge(
"save_long_term_memory",
END
)

# ============================================================

# Compile

# ============================================================

graph = builder.compile()
