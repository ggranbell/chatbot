"""LLM chain — builds LangChain prompts and invokes the chat model.

Two chains are exposed:
- ``plain_chain``  — direct chat (no retrieval context).
- ``rag_chain``    — answers grounded in retrieved context chunks.

Both use ``ChatOllama`` with a ``ChatPromptTemplate``.
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from rag.config import CHAT_MODEL, OLLAMA_BASE_URL

# ---------------------------------------------------------------------------
# Prompt templates — optimised for llama3.2:1b
#
# Design principles for small (≤3B) models:
# 1. Keep system prompt SHORT — every token eats into the context window.
# 2. Use simple, imperative sentences — fewer clauses = better compliance.
# 3. Put the strongest constraint first (only use context).
# 4. Give ONE example of the refusal phrase so the model knows the exact words.
# 5. Minimal formatting rules — the model can't follow many at once.
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """\
You are a strict document-based assistant. You MUST answer ONLY from the CONTEXT provided. NEVER use your own knowledge or training data.

Rules (follow every rule, no exceptions):
1. ONLY use facts, names, numbers, and details that appear in the CONTEXT below.
2. Do NOT add any information from your training data or general knowledge.
3. If the CONTEXT does not contain the answer, reply EXACTLY: "The provided context does not contain enough information to answer this question." Do not try to help further.
4. Do NOT guess, assume, or infer anything that is not explicitly stated in the CONTEXT.
5. If the user asks about something not in the CONTEXT, refuse — do not improvise.
6. If the CONTEXT contains information unrelated to the QUESTION, IGNORE it. ONLY use information that directly answers the QUESTION.
7. NEVER restate the question, never add extra commentary, and never speculate.
8. Only answer the question do not add any information whether irrelevant or relevant to the question.

Format:
- Use **bold** for key terms.
- Use bullet points for lists.
- Keep answers short, factual, and strictly relevant to the QUESTION.\
"""

RAG_HUMAN_TEMPLATE = """\
CONTEXT:
{context}

---
QUESTION: {question}

Answer:\
"""

RAG_HUMAN_TEMPLATE_WITH_HISTORY = """\
CONVERSATION HISTORY:
{history}

CONTEXT:
{context}

---
QUESTION: {question}

Answer:\
"""

PLAIN_SYSTEM_PROMPT = """\
You are a helpful assistant. Give clear, concise answers. Use Markdown formatting when appropriate.\
"""

PLAIN_HUMAN_TEMPLATE = "{question}"

# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _get_llm(model: str = CHAT_MODEL) -> ChatOllama:
    return ChatOllama(model=model, base_url=OLLAMA_BASE_URL, reasoning=False)


# ---------------------------------------------------------------------------
# Chain builders
# ---------------------------------------------------------------------------

def build_rag_chain(model: str = CHAT_MODEL, *, with_history: bool = False):
    """Return a runnable chain: (context, question[, history]) → str."""
    template = RAG_HUMAN_TEMPLATE_WITH_HISTORY if with_history else RAG_HUMAN_TEMPLATE
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", template),
    ])
    return prompt | _get_llm(model) | StrOutputParser()


def build_plain_chain(model: str = CHAT_MODEL):
    """Return a runnable chain: (question) → str."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", PLAIN_SYSTEM_PROMPT),
        ("human", PLAIN_HUMAN_TEMPLATE),
    ])
    return prompt | _get_llm(model) | StrOutputParser()


# ---------------------------------------------------------------------------
# Convenience invokers
# ---------------------------------------------------------------------------

def ask_plain(question: str, *, model: str = CHAT_MODEL) -> str:
    """Send a plain question (no RAG context) and return the answer."""
    chain = build_plain_chain(model)
    return chain.invoke({"question": question})


def ask_with_context(
    question: str,
    context: str,
    *,
    conversation_history: str = "",
    model: str = CHAT_MODEL,
) -> str:
    """Send a question grounded in *context* and return the answer.

    When *conversation_history* is non-empty it is injected into the prompt
    so the model can resolve follow-up references like "what about X?".
    """
    if conversation_history:
        chain = build_rag_chain(model, with_history=True)
        return chain.invoke({
            "context": context,
            "question": question,
            "history": conversation_history,
        })
    chain = build_rag_chain(model)
    return chain.invoke({"context": context, "question": question})
