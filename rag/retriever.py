import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

persist_dir = "index"
top_k = 5

def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    return vectordb

def retrieve_chunks(question: str, k: int = top_k):
    vectordb = load_vectorstore()
    results = vectordb.similarity_search_with_score(question, k=k)
    return results

def format_citations(results):
    citations = []
    for doc, _score in results:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "NA")
        citations.append(f"{source} (page{page})")
    return list(set(citations))

def answer_question(question: str):
    results = retrieve_chunks(question)
    if not results:
        return "I don't have an answer based on the provided documents.", []
    
    context = "\n\n".join([doc.page_content for doc, _ in results])
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )

    prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the provided context.
Do NOT use prior knowledge.
If the answer is not explicitly stated in the context, say:
"I don't have an answer based on the provided documents."

Context:
{context}

Question:
{question}

Answer concisely and factually.
"""
    response = llm.invoke(prompt)
    citations = format_citations(results)
    return response.content.strip(), citations

if __name__ == "__main__":
    question = "How is artificial intelligence used to improve healthcare operations?"
    answer, sources = answer_question(question)

    print("\nANSWER:\n", answer)
    print("\nSOURCES:")
    for s in sources:
        print("-", s)