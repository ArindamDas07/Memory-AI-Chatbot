-- ============================================================
-- ONCEMORE AI COMPANION DATABASE SCHEMA
-- Purpose: Support versioned Long-Term Memory and Episodic Memory
-- Author: Arindam Das
-- ============================================================

CREATE DATABASE IF NOT EXISTS oncemore;
USE oncemore;

-- 1. Sessions Table
-- Stores application-level conversation sessions.
CREATE TABLE sessions (
    session_id VARCHAR(100) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_chat_number INT NOT NULL DEFAULT 0
);

-- 2. Conversations Table
-- Stores each user message and the corresponding AI response.
CREATE TABLE conversations (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    chat_number INT NOT NULL,
    user_message TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE KEY unique_session_chat (session_id, chat_number),
    CONSTRAINT fk_session FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

-- 3. Conversation Embeddings Table
-- Stores vector embeddings for episodic memory (semantic retrieval).
CREATE TABLE conversation_embeddings (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    conversation_id BIGINT NOT NULL,
    session_id VARCHAR(100) NOT NULL,
    chat_number INT NOT NULL,
    embedding JSON NOT NULL, -- Stores the vector as a JSON array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
    UNIQUE KEY unique_conversation_embedding (conversation_id)
);

-- 4. Long-Term Memories Table
-- Core Memory Engine: Supports 'active' and 'superseded' facts to handle life changes.
CREATE TABLE long_term_memories (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    session_id VARCHAR(100) NOT NULL,
    
    -- Added during incremental improvement phase:
    memory_category VARCHAR(50) NOT NULL COMMENT 'IDENTITY, CAREER, TRAVEL, etc.',
    
    memory_key VARCHAR(150) NOT NULL,
    memory_value TEXT NOT NULL,

    -- Versioning System: Allows the bot to "forget" old facts while keeping history.
    status ENUM('active', 'superseded', 'retired') DEFAULT 'active',

    source_conversation_id BIGINT NOT NULL,
    source_chat_number INT NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    FOREIGN KEY (source_conversation_id) REFERENCES conversations(id),
    
    -- Optimized Indexes for Retrieval Nodes
    INDEX idx_user_status (user_id, status),
    INDEX idx_category (memory_category),
    INDEX idx_memory_key (memory_key)
);

-- ============================================================
-- OPTIONAL: RESET SCRIPT
-- Use these to clear data for a fresh 50-turn test run.
-- ============================================================
/*
SET FOREIGN_KEY_CHECKS = 0;
TRUNCATE TABLE long_term_memories;
TRUNCATE TABLE conversation_embeddings;
TRUNCATE TABLE conversations;
UPDATE sessions SET last_chat_number = 0;
SET FOREIGN_KEY_CHECKS = 1;
*/