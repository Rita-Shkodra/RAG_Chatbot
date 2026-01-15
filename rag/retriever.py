import os
from typing import List, Tuple
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PERSIST_DIR = "index"

TOP_K = 5
CANDIDATES = 20
LOW_CONFIDENCE = 0.6

LLM_MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0

INJECTION_PHRASES = [
    "ignore the context",
    "ignore previous instructions",
    "use your own knowledge",
    "pretend you are",
    "forget the documents",
]


def load_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )


def retrieve_candidates(question: str) -> List[Tuple]:
    vectordb = load_vectorstore()
    return vectordb.similarity_search_with_score(question, k=CANDIDATES)


def is_low_confidence(results: List[Tuple]) -> bool:
    if not results:
        return True
    best_score = min(score for _, score in results)
    return best_score > LOW_CONFIDENCE


def rerank_with_llm(question: str, results: List[Tuple]) -> List[Tuple]:
    if len(results) <= TOP_K:
        return results

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    numbered_chunks = []
    for i, (doc, _) in enumerate(results, start=1):
        snippet = doc.page_content[:600].replace("\n", " ")
        numbered_chunks.append(f"{i}. {snippet}")

    prompt = f"""
Question:
{question}

Excerpts:
{chr(10).join(numbered_chunks)}

Return ONLY the numbers of the {TOP_K} most relevant excerpts,
comma-separated.
"""

    response = llm.invoke(prompt).content.strip()

    try:
        indices = [
            int(i.strip()) - 1
            for i in response.split(",")
            if i.strip().isdigit()
        ]
        return [results[i] for i in indices[:TOP_K]]
    except Exception:
        return results[:TOP_K]


def is_prompt_injection(question: str) -> bool:
    q = question.lower()
    return any(phrase in q for phrase in INJECTION_PHRASES)


def format_citations(results: List[Tuple]) -> List[str]:
    citations = set()
    for doc, _ in results:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")

        if isinstance(page, int):
            page = page + 1
        else:
            page = "NA"

    citations.add(f"{source} (page {page})")

    return sorted(citations)


def answer_question(question: str):
    if is_prompt_injection(question):
        return "I can only answer questions using the provided documents.", []

    results = retrieve_candidates(question)

    if is_low_confidence(results):
        return "I don't have an answer based on the provided documents.", []

    results = rerank_with_llm(question, results)

    context = "\n\n".join(doc.page_content for doc, _ in results)

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
    )

    prompt = f"""
Use ONLY the context below.

If the answer is not explicitly stated, say:
"I don't have an answer based on the provided documents. I am only programmed to answer question on healthcare and related topics."

Context:
{context}

Question:
{question}

Answer in a single, well-structured paragraph that synthesizes
information from all relevant parts of the context. Do not list
separate points. Do not attribute sentences to specific sources.
"""

    response = llm.invoke(prompt).content.strip()

    if "i don't have an answer based on the provided documents" in response.lower():
        return response, []

    citations = format_citations(results)
    return response, citations


if __name__ == "__main__":
    q = "When was Kosovo declared independent?"
    answer, sources = answer_question(q)

    print("\nANSWER:\n", answer)
    print("\nSOURCES:")
    for s in sources:
        print("-", s)
