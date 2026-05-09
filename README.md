# Multi-RAG Agentic Handling hundreds of tools and multiple documents using LlamaIndex

---

## 1. Overview

An agentic RAG system that answers research questions across **11 academic
papers** and you can increase more or add your documents also. The agent selects the right retrieval strategy per query using a
custom LLM-based tool scorer — built using LlamaIndex.

---

## 2. Workflow

### 2.1 LLM & Embedding Setup

- Used **Gemini 3.1 Flash Lite** via Google AI Studio
- Embedded chunks locally via **Ollama `nomic-embed-text`** (768-dim)
- API keys loaded securely from `.env` via `python-dotenv`

```python
Settings.llm = GoogleGenAI(
    model="gemini-3.1-flash-lite-preview",
    api_key=os.getenv("google_api_key")
)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
```

---

### 2.2 ChromaDB for Persistent Vector Storage

Used **ChromaDB** as the vector store for each paper's `VectorStoreIndex`,
so vector_database persist in disk and at re-running the project it won't calculate
embeddings from the begining but reload the persisted memory:

Used `storage_context.persist()` for `SummaryIndex` (which stores raw text,
not vectors) so it also survives restarts without rebuilding:

**Example of storage folders structure**

```bash
chroma_db/
├── multi_docs_chroma_db_metagpt       ← paper 1 vectors
├── multi_docs_chroma_db_longlora      ← paper 2 vectors
└── ...                                ← one isolated collection per paper
```

---

### 2.3 Convert Indices to tools

Gave each paper a `VectorStoreIndex` and a `SummaryIndex` with a
`QueryEngine` as a tool in front of them so the LLM can pick the right retrieval
strategy per query — similarity search for specific questions, sequential
read for summarization — without the caller needing to decide:
This yields **22 tools** total across 11 papers (2 per paper).

---

### 2.4 LLM-Based Tool Scorer

Used `ObjectIndex` to embed all 22 tool descriptions and retrieve the
top-5 candidates, but it failed as each tool description is so similar by cosine similarity per query, so sent each tool
to the LLM individually to score it on two values —
1) how relevant it is **right now** and
2) how likely it will be needed in a **later reasoning step**

This function logic is my own-idea and If anyone wants to try it out please refer to this repository (function is pushed seprately as python script)

Used `llm.acomplete()` (LlamaIndex's async completion method) so the event
loop is released during each HTTP round-trip rather than blocking the thread,
keeping scoring responsive even for larger candidate sets.

---

### 2.5 Agent Construction

Used `AgentWorkflow` with memory injected at runtime via `ChatMemoryBuffer`
so the agent chat maintaining multi-turn conversation context:

```python
agent_workflow = AgentWorkflow.from_tools_or_functions(
    best_tools,
    llm=llm,
    system_prompt="Always use the provided tools. Do not rely on prior knowledge."
)
response = await agent_workflow.run(user_msg=query, memory=memory)
```

---

## 3. Architecture

```bash
User Query
     │
     ▼
ObjectIndex  →  top-5 tools by cosine similarity
     │
     ▼
LLM Scorer   →  now_score + later_score  →  top-3 tools
     │
     ▼
AgentWorkflow (Gemini 3.1 Flash Lite)
     ├── vector_tool_<paper>   →  ChromaDB  →  VectorStoreIndex
     └── summary_tool_<paper>  →  Disk      →  SummaryIndex
     │
     ▼
ChatMemoryBuffer  →  Final Answer
```

---

## 4. Tech Stack

| Component    | Technology                                          |
|--------------|-----------------------------------------------------|
| LLM          | Gemini 3.1 Flash Lite (Google AI Studio)            |
| Embeddings   | Ollama `nomic-embed-text` (local, 768-dim)          |
| Vector store | ChromaDB (persistent, per-document collections)     |
| Framework    | LlamaIndex 0.14.22                                   |
| Agent        | `AgentWorkflow`                                     |
| Memory       | `ChatMemoryBuffer` (runtime-injected)               |
| Routing      | `RouterQueryEngine`           |

---

## 5. Papers / Documents

| File                          | Paper                                              |
|-------------------------------|----------------------------------------------------|
| `metagpt.pdf`                 | MetaGPT: Meta Programming for Multi-Agent Collab   |
| `longlora.pdf`                | LongLoRA: Efficient Fine-tuning of Long-Context LLMs|
| `loftq.pdf`                   | LoftQ: LoRA-Fine-Tuning-Aware Quantization         |
| `swebench.pdf`                | SWE-bench: Can LLMs Resolve Real GitHub Issues?    |
| `selfrag.pdf`                 | Self-RAG: Learning to Retrieve, Generate, Critique |
| `zipformer.pdf`               | Zipformer: A Better Encoder for ASR                |
| `values.pdf`                  | Measuring the Values Encoded in LLMs               |
| `finetune_fair_diffusion.pdf` | Fine-tuning Diffusion Models for Fairness          |
| `knowledge_card.pdf`          | Knowledge Card: Filling LLMs' Knowledge Gaps       |
| `metra.pdf`                   | METRA: Scalable Unsupervised RL                    |
| `vr_mcl.pdf`                  | VR-MCL: Visual Representation Learning             |

---

## 6. Setup

```bash
git clone <repo>
cd <repo>

conda create -n agentic_env python=3.10
conda activate agentic_env
pip install -r requirements.txt

ollama pull nomic-embed-text
cp .env.sample .env
```

`.env.sample`:

```dotenv
google_api_key =
```

Get a free key at https://aistudio.google.com

---

## 7. Conclusion

This project goes beyond standard RAG by treating **tools themselves as
retrievable objects** and scoring them on **future reasoning utility** — not
just immediate relevance. The full stack runs on free-tier infrastructure
(Gemini API + local Ollama embeddings) while remaining architecturally sound
for production scaling.
