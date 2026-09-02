CREATE DATABASE IF NOT EXISTS oncemore;

USE oncemore;


-- ============================================================
-- Sessions
-- Stores application-level conversation sessions.
-- ============================================================

CREATE TABLE sessions (
    session_id VARCHAR(100) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_chat_number INT NOT NULL DEFAULT 0
);


-- ============================================================
-- Conversations
-- Stores each user message and corresponding AI response.
-- ============================================================

CREATE TABLE conversations (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    chat_number INT NOT NULL,
    user_message TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE KEY unique_session_chat (session_id, chat_number)
);


-- ============================================================
-- Conversation Embeddings
-- Stores vector embeddings for episodic memory retrieval.
-- ============================================================

CREATE TABLE conversation_embeddings (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    conversation_id BIGINT NOT NULL,
    session_id VARCHAR(100) NOT NULL,
    chat_number INT NOT NULL,
    embedding JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (conversation_id)
        REFERENCES conversations(id),

    UNIQUE KEY unique_conversation_embedding (conversation_id)
);


-- ============================================================
-- Long-Term Memories
-- Stores persistent user-specific memories.
-- ============================================================

CREATE TABLE long_term_memories (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,

    user_id VARCHAR(100) NOT NULL,
    session_id VARCHAR(100) NOT NULL,

    memory_key VARCHAR(150) NOT NULL,
    memory_value TEXT NOT NULL,

    status ENUM(
        'active',
        'superseded',
        'retired'
    ) DEFAULT 'active',

    source_conversation_id BIGINT NOT NULL,
    source_chat_number INT NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    FOREIGN KEY (source_conversation_id)
        REFERENCES conversations(id),

    INDEX idx_memory_key (memory_key),
    INDEX idx_session_status (session_id, status)
);