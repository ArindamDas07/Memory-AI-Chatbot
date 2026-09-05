import os
import uuid
import mysql.connector
from dotenv import load_dotenv


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# Database connection
# ============================================================

def get_connection():
    """Create and return a MySQL database connection."""

    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )




def create_session(connection):
    """Create a new session and initialize its chat counter."""

    session_id = f"session_{uuid.uuid4().hex[:16]}"

    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO sessions
            (session_id, last_chat_number)
            VALUES (%s, %s)
            """,
            (session_id, 0)
        )

        connection.commit()

        return session_id

    except Exception:

        connection.rollback()
        raise

    finally:

        cursor.close()



def get_next_chat_number(connection, session_id):
    """Return the next chat number for a session."""

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT last_chat_number
        FROM sessions
        WHERE session_id = %s
        """,
        (session_id,)
    )

    result = cursor.fetchone()

    if result is None:
        cursor.close()
        raise ValueError("Session does not exist")

    next_number = result[0] + 1

    cursor.close()

    return next_number

def save_chat(connection, session_id, user_message, ai_response):
    """Save a conversation and update the session chat counter."""

    cursor = connection.cursor()

    try:
        chat_number = get_next_chat_number(
            connection,
            session_id
        )

        cursor.execute(
            """
            INSERT INTO conversations
            (session_id, chat_number, user_message, ai_response)
            VALUES (%s, %s, %s, %s)
            """,
            (
                session_id,
                chat_number,
                user_message,
                ai_response
            )
        )

        # Capture immediately after INSERT
        conversation_id = cursor.lastrowid

        cursor.execute(
            """
            UPDATE sessions
            SET last_chat_number = %s
            WHERE session_id = %s
            """,
            (chat_number, session_id)
        )

        connection.commit()

        return conversation_id, chat_number

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()

def get_chat_history(connection, session_id):
    """Return the conversation history for a session."""

    cursor = connection.cursor(dictionary=True)

    try:

        cursor.execute(
            """
            SELECT
                id,
                session_id,
                chat_number,
                user_message,
                ai_response,
                created_at
            FROM conversations
            WHERE session_id = %s
            ORDER BY chat_number ASC
            """,
            (session_id,)
        )

        return cursor.fetchall()

    finally:

        cursor.close()