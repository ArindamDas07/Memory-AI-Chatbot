import json
import math
import ollama
import logging

from backend.llm import (
    generate_with_llm,
    memory_extractor_llm,
    memory_comparison_llm
)

from backend.state import CompanionState
from backend.database import (
    get_connection,
    save_chat
)

# Configuration for local embedding model
EMBEDDING_MODEL = "nomic-embed-text"

# Setup logging for production-grade error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# 1. Retrieval Nodes
# ============================================================

def retrieve_long_term_memory(state: CompanionState):
    """
    Retrieves the COMPLETE user biography including Active and Superseded facts.
    This historical context prevents hallucinations and allows temporal reasoning.
    """
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT id, memory_category, memory_key, memory_value, status, updated_at
            FROM long_term_memories
            WHERE user_id = %s
            ORDER BY updated_at DESC
            """,
            (state["user_id"],)
        )
        memories = cursor.fetchall()
        return {"long_term_memories": memories}
    finally:
        cursor.close()
        connection.close()

def retrieve_episodic_memory(state: CompanionState):
    """
    Performs a semantic vector search over the current session's interactions.
    Links the user message to the most relevant past conversation snippets.
    """
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=state["user_message"])
        query_emb = response["embeddings"][0]
        cursor.execute(
            "SELECT ce.embedding, c.user_message, c.ai_response FROM conversation_embeddings ce "
            "JOIN conversations c ON ce.conversation_id = c.id WHERE ce.session_id = %s", 
            (state["session_id"],)
        )
        rows = cursor.fetchall()
    finally:
        cursor.close()
        connection.close()

    memories = []
    for row in rows:
        stored_emb = json.loads(row["embedding"]) if isinstance(row["embedding"], str) else row["embedding"]
        sim = cosine_similarity(query_emb, stored_emb)
        if sim >= 0.60:
            memories.append({"user_message": row["user_message"], "ai_response": row["ai_response"], "similarity": sim})
    
    memories.sort(key=lambda x: x["similarity"], reverse=True)
    return {"episodic_memories": memories[:5]}

# ============================================================
# 2. Generation Node
# ============================================================

def generate_response(state: CompanionState):
    """
    Grounded Response Generation: Synthesizes LTM and Episodic memory into a 
    persona-consistent response. Strictly forbidden from imagining facts.
    """
    all_memories = state["long_term_memories"]
    episodic = state["episodic_memories"]

    # Construct the 'Biography' timeline for the prompt
    biography_parts = []
    for m in all_memories:
        label = "[CURRENT]" if m['status'] == 'active' else "[PAST/HISTORICAL]"
        biography_parts.append(f"{label} {m['memory_key']}: {m['memory_value']} (Updated: {m['updated_at']})")
    
    biography_text = "\n".join(biography_parts)

    episodic_text = "\n".join(
        f"- User: {m['user_message']}\n  Assistant: {m['ai_response']}"
        for m in episodic
    )

    prompt = f"""
You are a warm, consistent AI companion. 

STRICT PROTOCOL:
1. You only know the user via the 'USER BIOGRAPHY' and 'PAST CONVERSATIONS' below.
2. If info is NOT in the sections below, you MUST say "I don't recall you mentioning that" or ask the user.
3. DO NOT hallucinate companies, degrees, or names.
4. Use [CURRENT] facts for the present and [PAST] facts for history.

USER BIOGRAPHY:
{biography_text or "No facts known yet."}

PAST CONVERSATIONS:
{episodic_text or "No relevant past conversations."}

USER MESSAGE:
{state["user_message"]}

Respond naturally, warmly, and stay strictly grounded in the facts.
"""
    response = generate_with_llm(prompt)
    return {"ai_response": response}

# ============================================================
# 3. Extraction & Conflict Resolution Nodes
# ============================================================

def extract_memory(state: CompanionState):
    """
    Multi-Fact Extraction Node: Identifies a list of personal disclosures.
    Injects existing keys to maintain schema consistency across turns.
    """
    existing_keys = list(set([m['memory_key'] for m in state.get("long_term_memories", [])]))
    
    prompt = f"""
Identify ALL new personal facts about the user. 
STRICT RULES:
1. Only extract what is EXPLICITLY stated.
2. If multiple facts appear (age AND university), extract BOTH as separate items.
3. Use existing keys if applicable: {existing_keys}
4. If it's a general question, return an empty list.

USER MESSAGE: {state["user_message"]}
"""
    try:
        result = memory_extractor_llm.invoke(prompt)
        valid_candidates = [cand for cand in result.observations if cand.should_store and cand.memory_value]
        return {"memory_candidate": valid_candidates} 
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return {"memory_candidate": []}

def compare_candidate_memory(state: CompanionState):
    """
    Conflict Resolution Node (The Judge): Compares new facts against active LTM 
    to decide between SAME, NEW, or UPDATE (versioning).
    """
    candidates = state.get("memory_candidate", [])
    existing_memories = [m for m in state.get("long_term_memories", []) if m['status'] == 'active']
    decisions = []

    for candidate in candidates:
        best_match, best_score = None, 0.0

        for memory in existing_memories:
            match_text = f"{memory['memory_category']} {memory['memory_key']} {memory['memory_value']}"
            cand_text = f"{candidate.memory_category} {candidate.memory_key} {candidate.memory_value}"
            
            m_emb = ollama.embed(model=EMBEDDING_MODEL, input=match_text)["embeddings"][0]
            c_emb = ollama.embed(model=EMBEDDING_MODEL, input=cand_text)["embeddings"][0]
            score = cosine_similarity(m_emb, c_emb)
            
            if score > best_score:
                best_score, best_match = score, memory

        if best_match and best_score > 0.70:
            prompt = f"Existing: {best_match['memory_value']}\nNew: {candidate.memory_value}\nSAME or UPDATE?"
            try:
                res = memory_comparison_llm.invoke(prompt)
                decisions.append({"candidate": candidate, "decision": res.decision.upper(), "match": best_match})
            except:
                # Fallback for API/Tool call failures
                decisions.append({"candidate": candidate, "decision": "UPDATE", "match": best_match})
        else:
            decisions.append({"candidate": candidate, "decision": "NEW", "match": None})

    return {"memory_decision_list": decisions}

# ============================================================
# 4. Database Persistence Nodes
# ============================================================

def save_long_term_memory(state: CompanionState):
    """
    Atomic Persistence Node: Implements the versioning system by marking 
    conflicting facts as 'superseded' and inserting new 'active' truths.
    """
    decisions = state.get("memory_decision_list", [])
    if not decisions: return {}

    connection = get_connection()
    cursor = connection.cursor()

    try:
        for item in decisions:
            cand, decision, match = item["candidate"], item["decision"], item["match"]

            # NULL GUARD: Prevent database integrity crashes
            if not cand.memory_value or cand.memory_value.lower() == "none":
                continue

            if decision == "SAME": continue

            if decision == "UPDATE" and match:
                cursor.execute("UPDATE long_term_memories SET status = 'superseded' WHERE id = %s", (match["id"],))

            cursor.execute(
                """
                INSERT INTO long_term_memories 
                (user_id, session_id, memory_category, memory_key, memory_value, status, source_conversation_id, source_chat_number)
                VALUES (%s, %s, %s, %s, %s, 'active', %s, %s)
                """,
                (state["user_id"], state["session_id"], cand.memory_category, cand.memory_key, cand.memory_value, state["conversation_id"], state["chat_number"])
            )
        connection.commit()
        return {}
    except Exception as e:
        logger.error(f"Database Integrity Guard Triggered: {e}")
        connection.rollback()
        return {}
    finally:
        cursor.close()
        connection.close()

def save_conversation(state: CompanionState):
    connection = get_connection()
    try:
        id, num = save_chat(connection, state["session_id"], state["user_message"], state["ai_response"])
        return {"conversation_id": id, "chat_number": num}
    finally:
        connection.close()

def store_conversation_embedding(state: CompanionState):
    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(
            "INSERT INTO conversation_embeddings (conversation_id, session_id, chat_number, embedding) VALUES (%s, %s, %s, %s)",
            (state["conversation_id"], state["session_id"], state["chat_number"], json.dumps(state["conversation_embedding"]))
        )
        connection.commit()
        return {}
    except Exception as e:
        logger.error(f"Embedding Storage Failed: {e}")
        return {}
    finally:
        cursor.close()
        connection.close()

# ============================================================
# 5. Utilities
# ============================================================

def generate_conversation_embedding(state: CompanionState):
    """Bakes the 'Interaction Pair' (User+AI) into the vector for context retention."""
    text = f"User: {state['user_message']}\nAssistant: {state['ai_response']}"
    response = ollama.embed(model=EMBEDDING_MODEL, input=text)
    return {"conversation_embedding": response["embeddings"][0]}

def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot_product / (mag_a * mag_b) if mag_a and mag_b else 0.0