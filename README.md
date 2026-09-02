# 🧠 Memory AI Chatbot

A conversational AI chatbot with **episodic memory and long-term personal memory**, built using **LangGraph, Ollama, MySQL, and Streamlit**.

The goal of this project is to explore how a conversational AI system can maintain continuity across conversations instead of treating every user message as an isolated interaction.

The system combines:

* LLM-based response generation
* Semantic retrieval of previous conversations
* Persistent personal-memory extraction
* Semantic memory comparison
* Memory update/versioning
* Session-based conversation storage
* LangGraph workflow orchestration
* MySQL persistence
* Local LLM and embedding models through Ollama

---

## ✨ Features

### 1. Conversational AI

The chatbot uses Google's **Gemma 3 4B** model through Ollama to generate conversational responses.

The model is configured with:

```python
ChatOllama(
    model="gemma3:4b",
    temperature=0
)
```

The temperature is set to `0` to make the generation more deterministic.

---

### 2. Episodic Memory

The system stores every conversation together with its embedding.

For each conversation:

```text
User message
      +
Assistant response
      ↓
Embedding model
      ↓
Conversation embedding
      ↓
MySQL
```

The embedding model used is:

```text
nomic-embed-text
```

When the user sends a new message, the system generates an embedding for that message and compares it with previously stored conversation embeddings.

The comparison uses **cosine similarity**.

Only conversations above the configured similarity threshold are considered relevant.

Current configuration:

```text
Similarity threshold: 0.60
Maximum retrieved conversations: 5
```

The retrieved conversations are then provided to the generation LLM as relevant past conversations.

### Current scope

The current implementation performs episodic retrieval within the user's **current session**.

It does not yet use a dedicated vector database or database-native vector similarity search. Embeddings are stored as JSON in MySQL and similarity is calculated in Python.

---

## 🧠 Long-Term Memory

Episodic memory and long-term memory serve different purposes.

### Episodic memory

Episodic memory answers:

> "Have we talked about something similar before?"

For example:

```text
User:
I was asking about GPUs yesterday.
```

The system can retrieve a semantically similar previous conversation.

### Long-term memory

Long-term memory answers:

> "What persistent information do I know about this user?"

For example:

```text
User:
My name is Arindam Das.
```

The system can extract:

```text
memory_key:
user_name

memory_value:
user's name is Arindam Das
```

Another example:

```text
User:
I love to eat rice and dal.
```

The system can extract a persistent preference such as:

```text
memory_key:
food_preference

memory_value:
likes rice and dal
```

---

# 🔄 Long-Term Memory Strategy

The long-term memory pipeline follows three main stages.

## 1. Extract

The user's message is passed to a structured-output LLM.

The LLM determines whether the message contains a persistent personal fact.

The output follows the `MemoryCandidate` schema:

```text
should_store
memory_key
memory_value
reason
```

For example:

```text
User:
I want to learn Japanese.

↓

should_store = true
memory_key = learning_goal
memory_value = wants to learn Japanese
```

General questions are not stored.

For example:

```text
User:
What is the capital of India?

↓

should_store = false
```

---

## 2. Compare

If a memory candidate is produced, the system retrieves the user's existing active long-term memories.

The candidate and existing memories are converted into embeddings using:

```text
nomic-embed-text
```

The system calculates cosine similarity between the candidate and existing memories.

The most similar existing memory is selected.

If the similarity is below:

```text
0.60
```

the candidate is treated as a new memory.

If a sufficiently similar memory exists, the system uses the LLM to make the final semantic decision.

The comparison has two possible results:

```text
SAME
UPDATE
```

### SAME

The candidate represents the same underlying fact.

Example:

```text
Existing:
favorite_sport → likes football

Candidate:
favorite_sport → likes football
```

Result:

```text
SAME
```

Nothing is changed.

### UPDATE

The candidate changes or replaces the previous fact.

Example:

```text
Existing:
relocation_plan → plans to move to Dubai

Candidate:
relocation_plan → decided to stay in Kolkata
```

Result:

```text
UPDATE
```

The old memory is marked as:

```text
superseded
```

and a new active memory is inserted.

---

# 🗃️ Database Design

The project currently uses **MySQL**.

Database:

```text
oncemore
```

The database contains four main tables.

```text
sessions
    │
    └── conversations
            │
            └── conversation_embeddings

long_term_memories
    │
    └── source_conversation_id
            │
            └── conversations
```

---

## 1. `sessions`

Stores application sessions.

Important fields:

```text
session_id
created_at
last_chat_number
```

The session ID is generated when a new application session is created.

Example:

```text
session_5da877175f0b489d
```

The session also maintains the latest chat number.

---

## 2. `conversations`

Stores the actual conversation.

Important fields:

```text
id
session_id
chat_number
user_message
ai_response
created_at
```

Each conversation receives a unique database ID and a sequential chat number within the session.

A unique constraint is used on:

```text
(session_id, chat_number)
```

---

## 3. `conversation_embeddings`

Stores the embedding associated with each conversation.

Important fields:

```text
conversation_id
session_id
chat_number
embedding
created_at
```

The embedding is currently stored as JSON.

The relationship is:

```text
conversation
      │
      └── conversation_embedding
```

---

## 4. `long_term_memories`

Stores persistent personal information extracted from user messages.

Important fields include:

```text
id
user_id
session_id
memory_key
memory_value
status
source_conversation_id
source_chat_number
created_at
updated_at
```

Memory status can be:

```text
active
superseded
retired
```

This allows the system to preserve the history of changed memories instead of simply overwriting the old record.

For example:

```text
Old memory
status = superseded

New memory
status = active
```

The memory also keeps a reference to the conversation from which it originated.

---

# 🏗️ System Architecture

The application consists of four major layers.

```text
┌──────────────────────────────┐
│          Streamlit           │
│          Frontend            │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│          LangGraph           │
│       Workflow Engine        │
└──────────────┬───────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌─────────────┐  ┌─────────────┐
│    Ollama   │  │    MySQL    │
│             │  │             │
│ Gemma 3 4B  │  │ Conversations│
│             │  │ Embeddings  │
│ nomic-embed │  │ Memories    │
└─────────────┘  └─────────────┘
```

---

# 🔀 LangGraph Workflow

The complete conversation workflow is implemented using LangGraph.

The workflow is:

```text
START
  │
  ▼
retrieve_episodic_memory
  │
  ▼
retrieve_long_term_memory
  │
  ▼
generate_response
  │
  ▼
save_conversation
  │
  ▼
generate_conversation_embedding
  │
  ▼
store_conversation_embedding
  │
  ▼
extract_memory
  │
  ▼
compare_candidate_memory
  │
  ├────────────── SAME ──────────────┐
  │                                  │
  │                                  ▼
  │                                 END
  │
  ├────────────── None ─────────────┐
  │                                  │
  │                                  ▼
  │                                 END
  │
  └──────── NEW / UPDATE ──────────►
                     │
                     ▼
          save_long_term_memory
                     │
                     ▼
                    END
```

---

# 🔍 Step-by-Step Workflow

## Step 1 — Retrieve Episodic Memory

The current user message is converted into an embedding.

The system searches the stored conversation embeddings for the current session.

Cosine similarity is calculated:

```text
query embedding
       ↓
compare with stored embeddings
       ↓
similarity score
       ↓
rank
       ↓
top relevant conversations
```

The current implementation keeps conversations with:

```text
similarity >= 0.60
```

and limits the result to five conversations.

---

## Step 2 — Retrieve Long-Term Memory

The system retrieves active long-term memories belonging to the user.

Only memories with:

```text
status = active
```

are used.

These memories are later provided to the response-generation model.

---

## Step 3 — Generate Response

The Gemma 3 4B model receives:

```text
Persona
+
Relevant long-term memories
+
Relevant past conversations
+
Current user message
```

The LLM then generates the response.

The memory itself is not exposed to the user as a separate system.

---

## Step 4 — Save Conversation

The user message and generated AI response are stored in:

```text
conversations
```

A new:

```text
conversation_id
chat_number
```

is generated.

---

## Step 5 — Generate Conversation Embedding

The complete conversation is represented as:

```text
User: <user message>
Assistant: <assistant response>
```

This text is passed to:

```text
nomic-embed-text
```

to create the conversation embedding.

---

## Step 6 — Store Conversation Embedding

The generated embedding is stored in:

```text
conversation_embeddings
```

and linked to the corresponding conversation.

---

## Step 7 — Extract Long-Term Memory

The user's message is independently analyzed by the memory extraction LLM.

The purpose is not to remember everything.

Instead, the system attempts to identify information that is:

* personal
* persistent
* useful in future conversations

Examples include:

```text
Personal preferences
Goals
Plans
Hobbies
Career information
Important relationships
Persistent circumstances
Personal decisions
```

---

## Step 8 — Compare Candidate Memory

If the extractor determines that a memory should be stored, the system searches existing active memories.

The candidate is embedded and compared with existing memories.

The most similar memory is then evaluated semantically by the LLM.

The final decision is:

```text
SAME
```

or:

```text
UPDATE
```

If no sufficiently similar memory exists:

```text
NEW
```

is selected.

---

## Step 9 — Conditional Routing

LangGraph uses a conditional router after memory comparison.

```text
                 compare_candidate_memory
                           │
              ┌────────────┼─────────────┐
              │            │             │
             NEW         UPDATE         SAME
              │            │             │
              └──────┬─────┘             │
                     │                   END
                     ▼
           save_long_term_memory
                     │
                     ▼
                    END
```

This prevents unnecessary database writes when the candidate does not represent new information.

---

# 🧩 Project Structure

The intended repository structure is:

```text
Memory-AI-Chatbot/
│
├── backend/
│   ├── __init__.py
│   ├── database.py
│   ├── graph.py
│   ├── llm.py
│   ├── nodes.py
│   └── state.py
│
├── frontend/
│   └── app.py
│
├── database/
│   └── schema.sql
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 🤖 Models Used

## Generation Model

```text
Gemma 3 4B
```

Ollama model name:

```text
gemma3:4b
```

Used for:

* conversational response generation
* long-term memory extraction
* semantic memory comparison

---

## Embedding Model

```text
nomic-embed-text
```

Used for:

* conversation embeddings
* episodic memory retrieval
* long-term memory similarity comparison

---

# 🛠️ Technology Stack

| Component              | Technology             |
| ---------------------- | ---------------------- |
| Frontend               | Streamlit              |
| Workflow orchestration | LangGraph              |
| LLM                    | Gemma 3 4B             |
| LLM runtime            | Ollama                 |
| Embeddings             | nomic-embed-text       |
| Database               | MySQL                  |
| Database driver        | mysql-connector-python |
| Data validation        | Pydantic               |
| Configuration          | python-dotenv          |
| Language               | Python 3.12            |

---

# 🚀 Running the Project

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd Memory-AI-Chatbot
```

---

## 2. Create a Python 3.12 virtual environment

This project should be run using **Python 3.12**.

### Windows

```bash
py -3.12 -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3.12 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Install and run Ollama

Make sure Ollama is installed and running.

Pull the required models:

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

The application uses Ollama locally for both generation and embeddings.

---

## 5. Configure MySQL

Create the database:

```sql
CREATE DATABASE oncemore;
```

Then execute:

```text
database/schema.sql
```

This creates the required tables:

```text
sessions
conversations
conversation_embeddings
long_term_memories
```

---

## 6. Configure environment variables

Create a local `.env` file.

Example:

```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_mysql_user
DB_PASSWORD=your_mysql_password
DB_NAME=oncemore
```

Do **not** commit the `.env` file to GitHub.

Use `.env.example` for documenting the required configuration.

---

## 7. Start the application

From the project root:

```bash
python -m streamlit run frontend/app.py
```

The Streamlit frontend provides the conversational interface using its chat components.

---

# 💬 Example Interaction

The system can remember persistent information such as:

```text
User:
my name is arindam das

Assistant:
Nice to meet you, Arindam Das!
```

Later:

```text
User:
what is my name?
```

The system can use the stored long-term memory to answer consistently.

Similarly:

```text
User:
i love to eat rice and dal
```

can be extracted as a personal preference and stored as long-term memory.

A general question such as:

```text
what is the capital of india?
```

should not create a long-term personal memory because it does not contain a persistent fact about the user.

---

# 🎯 Design Philosophy

The main idea behind this project is that **conversation history and personal memory are not the same thing**.

Simply storing every previous message does not automatically create useful memory.

This project therefore separates memory into two concepts:

```text
                 User Message
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
       Episodic Memory    Long-Term Memory
             │                 │
      What did we talk      What do we know
      about previously?     about the user?
```

### Episodic memory

Stores conversational experiences.

```text
Conversation
    ↓
Embedding
    ↓
Semantic retrieval
```

### Long-term memory

Stores persistent user facts.

```text
User message
    ↓
Memory extraction
    ↓
Candidate memory
    ↓
Similarity search
    ↓
Semantic comparison
    ↓
NEW / SAME / UPDATE
```

This separation allows the system to retrieve relevant conversations without treating every conversation as a permanent personal fact.

---

# 🔐 Memory Versioning

Long-term memories are not simply overwritten.

When an existing fact changes, the old memory is marked:

```text
superseded
```

and the new memory becomes:

```text
active
```

For example:

```text
Before:

relocation_plan
→ plans to move to Dubai
status = active
```

Later:

```text
User:
I changed my plan. I will stay in Kolkata.
```

The system can produce:

```text
Old memory
status = superseded

New memory
relocation_plan
→ plans to stay in Kolkata
status = active
```

This provides a basic form of memory history/versioning.

---

# ⚠️ Current Limitations

This is currently a **working prototype**, not a production-ready memory infrastructure.

Current limitations include:

* Embeddings are stored as JSON in MySQL.
* Cosine similarity is calculated in Python.
* Episodic retrieval is currently session-scoped.
* The current demo uses a fixed `user_id` value.
* Memory retrieval currently loads active memories before selecting the relevant context.
* There is no dedicated vector database.
* There is no authentication system.
* There is no multi-user production deployment.
* There is no memory deletion/user-controlled memory management interface.
* There is no automated evaluation framework for memory accuracy.
* There is no streaming LLM response.
* There is no sophisticated memory ranking model beyond similarity and LLM comparison.

These are intentional areas for future development rather than claims of functionality that is not currently implemented.

---

# 🔮 Future Implementation

The next stage of the project would focus on making the memory architecture more scalable, reliable, and production-oriented.

## 1. Vector Database / Native Vector Search

Replace JSON-based embedding storage and Python-side brute-force comparison with a proper vector search system.

Possible direction:

```text
MySQL
   +
Vector Database
```

or a database with native vector capabilities.

This would make semantic retrieval more scalable as the number of conversations increases.

---

## 2. Better Episodic Retrieval

The current approach uses:

```text
embedding similarity
+
threshold
+
top 5
```

A future version could introduce:

```text
Semantic similarity
        +
Recency
        +
Chat number
        +
Session relevance
        ↓
Final ranking
```

This would allow recent and semantically relevant conversations to receive higher priority.

---

## 3. User-Level Memory

The current prototype separates:

```text
user_id
session_id
```

but the demo currently uses:

```text
demo_user
```

A future version would introduce real user identity and allow the same user's long-term memory to persist across multiple sessions.

```text
User
 │
 ├── Session 1
 │      ├── Chat 1
 │      ├── Chat 2
 │      └── ...
 │
 ├── Session 2
 │      ├── Chat 1
 │      └── ...
 │
 └── Long-Term Memories
```

---

## 4. Memory Management

Future versions could allow users to:

```text
View memories
Edit memories
Delete memories
Retire memories
Disable memory
```

This would give the user direct control over what the system remembers.

---

## 5. Memory Evaluation

A proper evaluation dataset could be created to measure:

```text
Memory extraction accuracy
Memory retrieval accuracy
False memory rate
Memory update accuracy
Memory relevance
Response consistency
```

This would make it possible to quantitatively evaluate the memory architecture instead of relying only on manual testing.

---

## 6. Improved Memory Ranking

The current system primarily relies on embedding similarity.

A future ranking pipeline could combine:

```text
Semantic similarity
+
Recency
+
Conversation importance
+
Memory confidence
+
User preference
```

to determine which memories should enter the LLM context.

---

## 7. Production Architecture

The prototype could eventually be separated into independent services:

```text
                ┌───────────────┐
                │    Frontend   │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │   API Layer   │
                └───────┬───────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
   ┌────────────┐ ┌────────────┐ ┌────────────┐
   │ Conversation│ │   Memory   │ │ Retrieval  │
   │   Service   │ │  Service   │ │  Service   │
   └────────────┘ └────────────┘ └────────────┘
          │             │             │
          └─────────────┼─────────────┘
                        ▼
                ┌───────────────┐
                │ Data / Vector │
                │    Storage    │
                └───────────────┘
```

The current LangGraph implementation provides the foundation for separating these responsibilities into explicit workflow stages.

---

# 📌 Project Status

**Current status: Working prototype**

Implemented:

* [x] Streamlit conversational interface
* [x] Session creation
* [x] Conversation persistence
* [x] Chat numbering
* [x] Conversation embeddings
* [x] Episodic semantic retrieval
* [x] Long-term memory extraction
* [x] Structured memory output
* [x] Memory similarity comparison
* [x] SAME / UPDATE decision
* [x] NEW memory creation
* [x] Memory superseding
* [x] LangGraph workflow
* [x] Conditional memory routing
* [x] Local LLM inference with Ollama
* [x] Local embedding generation with Ollama
* [x] MySQL persistence

Planned:

* [ ] Production user authentication
* [ ] Cross-session user memory
* [ ] Vector database / native vector search
* [ ] Advanced memory ranking
* [ ] Recency-aware retrieval
* [ ] Memory management UI
* [ ] Memory evaluation framework
* [ ] Automated testing
* [ ] API layer
* [ ] Production deployment
* [ ] Observability and monitoring

---

# 📜 License

This project is currently intended as a learning and portfolio project.
