import streamlit as st
import time
from backend.database import (
    get_connection,
    create_session,
    get_chat_history,
)
from backend.graph import graph

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Memory AI Companion",
    page_icon="🧠",
    layout="centered"
)

# ============================================================
# Sidebar & Session Management
# ============================================================
with st.sidebar:
    st.title("Settings")
    st.markdown("---")
    if st.button("🗑️ Start New Session"):
        # Clear all state and URL parameters to start fresh
        st.session_state.clear()
        st.query_params.clear()
        st.rerun()
    
    st.markdown("### Architecture Details")
    st.info(
        "This AI utilizes a versioned Long-Term Memory (LTM) system. "
        "It tracks facts as [CURRENT] or [PAST] to maintain persona consistency "
        "across hundreds of turns."
    )

# ============================================================
# Application Session Initialization
# ============================================================
if "session_id" not in st.session_state:
    # Attempt to recover session_id from the URL (useful for browser refreshes)
    session_id = st.query_params.get("session_id")

    if not session_id:
        connection = get_connection()
        try:
            # Create a brand new session in MySQL
            session_id = create_session(connection)
        finally:
            connection.close()

        # Persist the new session_id in the URL
        st.query_params["session_id"] = session_id

    st.session_state.session_id = session_id

# ============================================================
# Load Conversation History
# ============================================================
if "messages" not in st.session_state:
    connection = get_connection()
    try:
        # Fetch all previous messages for this session from the database
        history = get_chat_history(connection, st.session_state.session_id)
    finally:
        connection.close()

    messages = []
    for row in history:
        messages.append({"role": "user", "content": row["user_message"]})
        messages.append({"role": "assistant", "content": row["ai_response"]})
    st.session_state.messages = messages

# ============================================================
# UI Display
# ============================================================
st.title("🧠 Memory AI")
st.caption(f"Connected to Session: {st.session_state.session_id}")

# Display chat history with Automatic Visual Turn Numbering
# Note: Turn numbering is UI-only to keep the LLM input clean.
turn_counter = 1

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.caption(f"Turn #{turn_counter}")
            turn_counter += 1
        st.write(message["content"])

# ============================================================
# Chat Input & Logic Execution
# ============================================================
user_message = st.chat_input("Type a message or share a fact...")

if user_message:
    # 1. Track the current turn number locally
    current_turn = turn_counter 

    # 2. Immediately display the user's message in the UI
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.caption(f"Turn #{current_turn}")
        st.write(user_message)

    # 3. Construct the LangGraph State 
    # (Matches the schema defined in backend/state.py)
    state = {
        "user_id": "demo_user",
        "session_id": st.session_state.session_id,
        "user_message": user_message,
        "ai_response": "",
        "conversation_id": 0,
        "chat_number": 0,
        "episodic_memories": [],
        "long_term_memories": [],
        "context": "",
        "conversation_embedding": [],
        "memory_candidate": [],        # Initialized as a list for multi-fact extraction
        "memory_match": None,
        "memory_decision": None,
        "memory_decision_list": []     # For tracking multiple conflict decisions
    }

    # 4. Invoke the Cognitive Graph
    with st.spinner("Consulting memory..."):
        try:
            result = graph.invoke(state)
            ai_response = result["ai_response"]
        except Exception as e:
            ai_response = "I encountered a hiccup with my memory. Could you say that again?"
            st.error(f"System Error: {e}")

    # 5. Display and persist the AI's response
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.write(ai_response)