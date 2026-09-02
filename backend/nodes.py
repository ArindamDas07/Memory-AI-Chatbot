import json
import math

import ollama

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


EMBEDDING_MODEL = "nomic-embed-text"



def retrieve_long_term_memory(state: CompanionState):
    """Retrieve active long-term memories for the current user."""

    connection = get_connection()

    cursor = connection.cursor(dictionary=True)

    try:

        cursor.execute(
            """
            SELECT
                id,
                user_id,
                memory_key,
                memory_value,
                status,
                source_conversation_id,
                source_chat_number
            FROM long_term_memories
            WHERE user_id = %s
              AND status = 'active'
            ORDER BY updated_at DESC
            """,
            (state["user_id"],)
        )

        memories = cursor.fetchall()

        return {
            "long_term_memories": memories
        }

    finally:

        cursor.close()
        connection.close()



def generate_response(state: CompanionState):
    """Generate a response using long-term and episodic memory."""

    long_term = state["long_term_memories"]
    episodic = state["episodic_memories"]

    long_term_text = "\n".join(
        f"- {m['memory_key']}: {m['memory_value']}"
        for m in long_term
    )

    episodic_text = "\n".join(
        f"- User: {m['user_message']}\n"
        f"  Assistant: {m['ai_response']}"
        for m in episodic
    )

    prompt = f"""
You are a warm, consistent AI companion.

PERSONA:
- Warm and supportive
- Conversational and natural
- Do not pretend to know things that are not provided
- Do not mention the memory system
- Do not dump memories unnecessarily
- Stay consistent with your own persona

RELEVANT LONG-TERM MEMORY:
{long_term_text or "None"}

RELEVANT PAST CONVERSATIONS:
{episodic_text or "None"}

USER:
{state["user_message"]}

Respond naturally to the user.
"""

    response = generate_with_llm(prompt)

    return {
        "ai_response": response
    }

def save_conversation(state: CompanionState):
    """Save the current conversation and assign its chat number."""

    connection = get_connection()

    try:
        conversation_id, chat_number = save_chat(
            connection,
            state["session_id"],
            state["user_message"],
            state["ai_response"]
        )

        return {
            "conversation_id": conversation_id,
            "chat_number": chat_number
        }

    finally:
        connection.close()





def generate_conversation_embedding(state: CompanionState):
    """Generate an embedding for the current conversation."""

    text = (
        f"User: {state['user_message']}\n"
        f"Assistant: {state['ai_response']}"
    )

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=text
    )

    embedding = response["embeddings"][0]

    return {
        "conversation_embedding": embedding
    }







def store_conversation_embedding(state: CompanionState):
    """Store the generated conversation embedding in the database."""

    connection = get_connection()
    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO conversation_embeddings
            (
                conversation_id,
                session_id,
                chat_number,
                embedding
            )
            VALUES (%s, %s, %s, %s)
            """,
            (
                state["conversation_id"],
                state["session_id"],
                state["chat_number"],
                json.dumps(state["conversation_embedding"])
            )
        )

        connection.commit()

        return {}

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""

    if len(a) != len(b):
        raise ValueError(
            "Vectors must have the same dimension"
        )

    dot_product = sum(
        x * y
        for x, y in zip(a, b)
    )

    magnitude_a = math.sqrt(
        sum(x * x for x in a)
    )

    magnitude_b = math.sqrt(
        sum(y * y for y in b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (
        magnitude_a * magnitude_b
    )        


def retrieve_episodic_memory(state: CompanionState):
    """Retrieve relevant past conversations using embedding similarity."""

    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        # ----------------------------------------------------
        # 1. Create embedding for current user message
        # ----------------------------------------------------

        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=state["user_message"]
        )

        query_embedding = response["embeddings"][0]

        # ----------------------------------------------------
        # 2. Get conversations for this user/session
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                ce.conversation_id,
                ce.session_id,
                ce.chat_number,
                ce.embedding,
                c.user_message,
                c.ai_response
            FROM conversation_embeddings ce
            JOIN conversations c
                ON ce.conversation_id = c.id
            WHERE ce.session_id = %s
            """,
            (state["session_id"],)
        )

        rows = cursor.fetchall()

    finally:
        cursor.close()
        connection.close()

    # --------------------------------------------------------
    # 3. Compare query embedding with stored embeddings
    # --------------------------------------------------------

    memories = []

    for row in rows:

        stored_embedding = row["embedding"]

        if isinstance(stored_embedding, str):
            stored_embedding = json.loads(stored_embedding)

        similarity = cosine_similarity(
            query_embedding,
            stored_embedding
)

        memories.append(
            {
                "conversation_id": row["conversation_id"],
                "session_id": row["session_id"],
                "chat_number": row["chat_number"],
                "user_message": row["user_message"],
                "ai_response": row["ai_response"],
                "similarity": similarity
            }
        )

    # --------------------------------------------------------
    # 4. Rank by similarity
    # --------------------------------------------------------

    memories.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    # --------------------------------------------------------
    # 5. Keep relevant memories
    # --------------------------------------------------------

    episodic_memories = [
        memory
        for memory in memories
        if memory["similarity"] >= 0.60
    ]

    # --------------------------------------------------------
    # 6. Limit number of memories
    # --------------------------------------------------------

    episodic_memories = episodic_memories[:5]

    print("\nEpisodic memories:")

    for memory in episodic_memories:
        print(
            {
                "conversation_id": memory["conversation_id"],
                "chat_number": memory["chat_number"],
                "similarity": memory["similarity"]
            }
        )

    return {
        "episodic_memories": episodic_memories
    }


def extract_memory(state: CompanionState):
    """Extract persistent personal facts from the user's message."""

    prompt = f"""
You are a long-term memory extraction system for an AI companion.

Your job is ONLY to identify persistent PERSONAL FACTS about the USER.

IMPORTANT:
A fact should be stored only if it tells us something about the user
that could reasonably be useful in a future conversation.

Store:
- the user's personal preferences
- the user's goals
- the user's plans
- the user's hobbies
- the user's career or work situation
- important relationships mentioned by the user
- persistent personal circumstances
- decisions the user has made about their own life

Do NOT store:
- general knowledge
- facts about countries, cities, people, history, science, etc.
- answers to questions
- information supplied only as an example
- information that is not about the user
- temporary conversational statements
- greetings or small talk
- facts about the assistant

CRITICAL RULE:

If the user's message does NOT contain a persistent personal fact,
set:

should_store = false

and set:

memory_key = null
memory_value = null

Examples:

USER:
"What is the capital of Japan?"

RESULT:
should_store = false

USER:
"Tell me about Dubai."

RESULT:
should_store = false

USER:
"I have decided to move to Dubai next year."

RESULT:
should_store = true
memory_key = "future_move"
memory_value = "planned relocation to Dubai next year"

USER:
"I want to learn Japanese."

RESULT:
should_store = true
memory_key = "learning_goal"
memory_value = "wants to learn Japanese"

USER:
"I like football."

RESULT:
should_store = true
memory_key = "favorite_sport"
memory_value = "likes football"

USER:
"I am just tired today."

RESULT:
should_store = false

Analyze ONLY this user's message:

USER MESSAGE:
{state["user_message"]}
"""

    result = memory_extractor_llm.invoke(prompt)

    print("\nMemory extraction:")
    print(result)

    return {
        "memory_candidate": result
    }

def save_long_term_memory(state: CompanionState):
    """Create a new memory or supersede an existing memory."""

    candidate = state["memory_candidate"]
    decision = state["memory_decision"]
    existing = state["memory_match"]

    # --------------------------------------------------------
    # Nothing to save
    # --------------------------------------------------------

    if candidate is None:
        return {}

    # --------------------------------------------------------
    # SAME → nothing to change
    # --------------------------------------------------------

    if decision == "SAME":
        return {}

    connection = get_connection()
    cursor = connection.cursor()

    try:

        # ----------------------------------------------------
        # NEW
        # ----------------------------------------------------

        if decision == "NEW":

            cursor.execute(
                """
                INSERT INTO long_term_memories
                (
                    user_id,
                    session_id,
                    memory_key,
                    memory_value,
                    status,
                    source_conversation_id,
                    source_chat_number
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    state["user_id"],
                    state["session_id"],
                    candidate.memory_key,
                    candidate.memory_value,
                    "active",
                    state["conversation_id"],
                    state["chat_number"]
                )
            )

        # ----------------------------------------------------
        # UPDATE
        # ----------------------------------------------------

        elif decision == "UPDATE":

            cursor.execute(
                """
                UPDATE long_term_memories
                SET status = 'superseded'
                WHERE id = %s
                  AND status = 'active'
                """,
                (existing["id"],)
            )

            cursor.execute(
                """
                INSERT INTO long_term_memories
                (
                    user_id,
                    session_id,
                    memory_key,
                    memory_value,
                    status,
                    source_conversation_id,
                    source_chat_number
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    state["user_id"],
                    state["session_id"],
                    candidate.memory_key,
                    candidate.memory_value,
                    "active",
                    state["conversation_id"],
                    state["chat_number"]
                )
            )

        else:

            raise ValueError(
                f"Unknown memory decision: {decision}"
            )

        connection.commit()

        return {}

    except Exception:

        connection.rollback()
        raise

    finally:

        cursor.close()
        connection.close()


def compare_candidate_memory(state: CompanionState):
    """Compare a memory candidate with existing user memories."""

    candidate = state["memory_candidate"]

    # --------------------------------------------------------
    # 1. No memory candidate
    # --------------------------------------------------------

    if candidate is None:

        return {
            "memory_decision": None,
            "memory_match": None
        }

    # --------------------------------------------------------
    # 2. Candidate exists but should not be stored
    # --------------------------------------------------------

    if not candidate.should_store:

        return {
            "memory_decision": None,
            "memory_match": None
        }

    # --------------------------------------------------------
    # 3. Get existing active memories
    # --------------------------------------------------------

    connection = get_connection()

    try:

        existing_memories = get_active_memories(
            connection,
            state["user_id"]
        )

    finally:

        connection.close()

    # --------------------------------------------------------
    # 4. No existing memories
    # --------------------------------------------------------

    if not existing_memories:

        print("\nMemory comparison:")
        print("No existing memories.")
        print("Decision: NEW")

        return {
            "memory_decision": "NEW",
            "memory_match": None
        }

    # --------------------------------------------------------
    # 5. Create embedding for candidate
    # --------------------------------------------------------

    candidate_text = (
        f"{candidate.memory_key}: "
        f"{candidate.memory_value}"
    )

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=candidate_text
    )

    candidate_embedding = response["embeddings"][0]

    # --------------------------------------------------------
    # 6. Find most similar existing memory
    # --------------------------------------------------------

    best_match = None
    best_score = 0.0

    for memory in existing_memories:

        memory_text = (
            f"{memory['memory_key']}: "
            f"{memory['memory_value']}"
        )

        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=memory_text
        )

        memory_embedding = response["embeddings"][0]

        score = cosine_similarity(
            candidate_embedding,
            memory_embedding
        )

        if score > best_score:

            best_score = score
            best_match = memory

    # --------------------------------------------------------
    # 7. Debug information
    # --------------------------------------------------------

    print("\nMemory comparison search:")

    print(
        "Candidate:",
        candidate.memory_key,
        "→",
        candidate.memory_value
    )

    if best_match:

        print(
            "Best match:",
            best_match["memory_key"],
            "→",
            best_match["memory_value"]
        )

        print(
            "Similarity:",
            best_score
        )

    # --------------------------------------------------------
    # 8. Similarity threshold
    # --------------------------------------------------------

    if best_match is None or best_score < 0.60:

        print("Decision: NEW")

        return {
            "memory_decision": "NEW",
            "memory_match": None
        }

    # --------------------------------------------------------
    # 9. Semantic comparison using LLM
    # --------------------------------------------------------

    comparison = compare_memory(
        candidate,
        best_match
    )

    # --------------------------------------------------------
    # 10. Debug LLM decision
    # --------------------------------------------------------

    print("\nLLM memory comparison:")

    print(
        "Decision:",
        comparison.decision
    )

    print(
        "Reason:",
        comparison.reason
    )

    # --------------------------------------------------------
    # 11. Return SAME / UPDATE
    # --------------------------------------------------------

    return {
        "memory_decision": comparison.decision,
        "memory_match": best_match
    }

def get_active_memories(connection, user_id):
    """Return all active long-term memories for a user."""

    cursor = connection.cursor(dictionary=True)

    try:

        cursor.execute(
            """
            SELECT
                id,
                user_id,
                memory_key,
                memory_value,
                status,
                source_conversation_id,
                source_chat_number
            FROM long_term_memories
            WHERE user_id = %s
              AND status = 'active'
            ORDER BY updated_at DESC
            """,
            (user_id,)
        )

        return cursor.fetchall()

    finally:

        cursor.close()

def compare_memory(candidate, existing_memory):
    """Use the LLM to decide whether a memory is unchanged or updated."""

    prompt = f"""
You are comparing two long-term memories about the SAME user.

Determine whether the candidate memory represents:

SAME:
The candidate expresses the same underlying fact as the existing memory.
The wording or memory key may be different.

UPDATE:
The candidate changes, replaces, or supersedes the existing fact.

IMPORTANT:

Do NOT choose UPDATE merely because the wording or memory_key is different.

Return ONLY one of these decisions:

SAME
UPDATE

EXISTING MEMORY:
Key: {existing_memory["memory_key"]}
Value: {existing_memory["memory_value"]}

CANDIDATE MEMORY:
Key: {candidate.memory_key}
Value: {candidate.memory_value}

Example 1:

Existing:
relocation_plan → Dubai next year

Candidate:
future_move → planned relocation to Dubai next year

Decision:
SAME

Example 2:

Existing:
relocation_plan → Dubai next year

Candidate:
relocation_plan → staying in Kolkata instead of moving to Dubai

Decision:
UPDATE

Example 3:

Existing:
favorite_sport → football

Candidate:
favorite_sport → football

Decision:
SAME
"""

    result = memory_comparison_llm.invoke(prompt)

    decision = result.decision.strip().upper()

    if decision not in ("SAME", "UPDATE"):
        raise ValueError(
            f"Invalid memory comparison decision: {decision}"
        )

    return result        