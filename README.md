# 🧠 Cognitive AI Companion: Long-Term Memory Architecture

This project implements a production-grade memory architecture for AI companions, specifically designed to solve the critical industry failures of **persona drift** and **temporal inconsistency**.

Developed as a "Founding Engineer" technical solution for **OnceMore**, this system moves beyond simple context-window history. It implements a **Versioned Long-Term Memory (LTM)** system that manages the "state" of a user's life (career changes, location moves, evolving preferences) over long durations.

## 🏗️ System Overview

The architecture is designed as a **Cognitive State Machine**. It treats the LLM as a reasoning engine while using Python and MySQL to enforce deterministic rules.

```
graph TD
    User((User)) --> Streamlit[Streamlit Frontend]
    Streamlit --> LG[LangGraph Orchestrator]
    
    subgraph "Reasoning & Storage Layer"
        LG <--> Groq[Groq: openai/gpt-oss-120b]
        LG <--> Ollama[Ollama: nomic-embed]
        LG <--> MySQL[(MySQL: Versioned LTM)]
    end

    style Groq fill:#f96,stroke:#333,stroke-width:2px
    style MySQL fill:#69f,stroke:#333,stroke-width:2px
    style LG fill:#9f6,stroke:#333,stroke-width:4px

```

1. The "Biography" Engine (Versioning)
   Most chatbots fail when facts change. This system treats memory as a versioned database:
   Active vs. Superseded: Facts are tagged with a status. When a conflict is detected (e.g., switching from Google to Tesla), the old truth is marked as superseded (historical) and the new one as active (current).
   Chronological Injection: The AI receives the user's entire history labeled as \[CURRENT\] or \[PAST/HISTORICAL\], allowing it to answer "Where did I work before?" with 100% accuracy.

2. Multi-Fact List Extraction
   Using the Groq openai/gpt-oss-120b model, the system utilizes a List-Based Extraction Pipeline. It can identify and save multiple distinct facts in a single turn (e.g., age and university) without information loss.

🔀 LangGraph Workflow
The conversation flow is orchestrated via a directed graph to ensure data integrity at every step.

```
flowchart TD
    Start((START)) --> RetrieveE[Retrieve Episodic Memory]
    RetrieveE --> RetrieveL[Retrieve LTM Biography]
    RetrieveL --> Gen[Generate Grounded Response]
    Gen --> SaveC[Save Conversation Turn]
    SaveC --> GenEmb[Generate Turn Embedding]
    GenEmb --> StoreEmb[Store Turn Embedding]
    StoreEmb --> Extract[Extract Observations List]
    Extract --> Judge[Compare & Resolve Conflicts]
    
    Judge --> Decision{Decision List}
    Decision -- "NEW or UPDATE" --> Persist[Save LTM & Supersede Old]
    Decision -- "SAME or Empty" --> End((END))
    Persist --> End

```

🚀 Key Engineering Innovations

Strict Grounding Protocol: The generation prompt forbids the AI from imagining details. If information isn't in the Biography, the bot is instructed to ask the user.
Deterministic Reliability Guards: Python-level validation gates ensure that if the LLM returns a Null or invalid value, the system logs the event and skips the save rather than crashing the database.
Context-Aware Keys: The Extractor is shown existing memory keys to ensure "Workplace" doesn't accidentally get saved as "Job" or "Company" in separate turns.

📊 Evaluation & Stress Test
The system's reliability was validated through a rigorous two-stage process:

1. 46-Turn Stress Test
   A continuous 46-turn conversation was conducted covering career transitions, cancelled travel plans, and shifting beverage preferences. The AI maintained a consistent persona and successfully resolved all information conflicts.

2. 10-Question Recall Audit
   An automated Evaluation Harness (LLM-as-a-Judge) was used to test the bot's recall after the 46-turn test.

| \# | Question | Expected Fact | Result | 
 | ----- | ----- | ----- | ----- | 
| 1 | Full Name | Arindam Das | ✅ PASS | 
| 2 | Current Work | Tesla | ✅ PASS | 
| 3 | Previous Work | Google | ✅ PASS | 
| 4 | Dog's Name | Leo | ✅ PASS | 
| 5 | Master's Degree | Jadavpur University | ✅ PASS | 
| 6 | Morning Drink | Coffee (formerly Tea) | ✅ PASS | 
| 7 | Dubai Status | Cancelled | ✅ PASS | 
| 8 | 2027 Plans | Japan | ✅ PASS | 
| 9 | Location | West Bengal / Kolkata | ✅ PASS | 
| 10 | Favorite Hobby | Photography | ✅ PASS | 

Overall Accuracy: 100% Detailed logs and audit trail are available in  **\*\****evaluation_report.ipynb***\*\****.*

🛠️ Tech Stack

Orchestration: LangGraph (StateGraph)
Reasoning Model: Groq openai/gpt-oss-120b
Embedding Model: Ollama nomic-embed-text
Database: MySQL 8.0 (Relational Versioning)
Frontend: Streamlit (Visual Turn-Numbering)
Language: Python 3.12

🚀 Setup & Installation

1. Database Setup
   Execute the consolidated schema located at database/schema.sql:

```
mysql -u your_user -p < database/schema.sql

```

2. Environment Setup
   Create a .env file in the root directory:

```
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_user
DB_PASSWORD=your_password
DB_NAME=oncemore
GROQ_API_KEY=your_groq_api_key

```

3. Run

```
pip install -r requirements.txt
python -m streamlit run frontend/app.py

```

🔐 Design Philosophy
Reasoning vs. Enforcement: LLMs are probabilistic; databases are deterministic. This project uses the LLM to understand the meaning of a change, but uses structured Python logic and SQL constraints to enforce that change. This ensures the AI remains a reliable companion that truly "grows" with the user.

Author: Arindam Das (M.E., Jadavpur University)
