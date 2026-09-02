import streamlit as st

from backend.database import (
    get_connection,
    create_session,
    get_chat_history,
)

from backend.graph import graph


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Memory AI",
    page_icon="🧠",
    layout="centered"
)


# ============================================================
# Application session
# ============================================================

if "session_id" not in st.session_state:

    # --------------------------------------------------------
    # Try to recover session_id from URL
    # --------------------------------------------------------

    session_id = st.query_params.get("session_id")

    # --------------------------------------------------------
    # No session_id -> create a new application session
    # --------------------------------------------------------

    if not session_id:

        connection = get_connection()

        try:

            session_id = create_session(
                connection
            )

        finally:

            connection.close()

        # ----------------------------------------------------
        # Persist session_id in URL
        # ----------------------------------------------------

        st.query_params["session_id"] = session_id

    # --------------------------------------------------------
    # Store application session in Streamlit state
    # --------------------------------------------------------

    st.session_state.session_id = session_id


# ============================================================
# Load conversation history
# ============================================================

if "messages" not in st.session_state:

    connection = get_connection()

    try:

        history = get_chat_history(
            connection,
            st.session_state.session_id
        )

    finally:

        connection.close()

    messages = []

    for row in history:

        messages.append(
            {
                "role": "user",
                "content": row["user_message"]
            }
        )

        messages.append(
            {
                "role": "assistant",
                "content": row["ai_response"]
            }
        )

    st.session_state.messages = messages


# ============================================================
# Title
# ============================================================

st.title("🧠 Memory AI")

st.caption(
    "A conversational AI with episodic and long-term memory"
)


# ============================================================
# Display conversation
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(
            message["content"]
        )


# ============================================================
# Chat input
# ============================================================

user_message = st.chat_input(
    "Type your message..."
)


if user_message:

    # --------------------------------------------------------
    # Display user message immediately
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    with st.chat_message("user"):

        st.write(user_message)


    # --------------------------------------------------------
    # Build graph state
    # --------------------------------------------------------

    state = {

        "user_id": "demo_user",

        "session_id":
            st.session_state.session_id,

        "user_message":
            user_message,

        "ai_response": "",

        "conversation_id": 0,

        "chat_number": 0,

        "episodic_memories": [],

        "long_term_memories": [],

        "context": "",

        "conversation_embedding": [],

        "memory_candidate": None,

        "memory_match": None,

        "memory_decision": None,
    }


    # --------------------------------------------------------
    # Run LangGraph
    # --------------------------------------------------------

    result = graph.invoke(state)


    # --------------------------------------------------------
    # AI response
    # --------------------------------------------------------

    ai_response = result["ai_response"]


    # --------------------------------------------------------
    # Display AI response
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": ai_response
        }
    )

    with st.chat_message("assistant"):

        st.write(ai_response)