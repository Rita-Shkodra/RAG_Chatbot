import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
data = "data/raw/healthcare"
persist_dir = "index"

def load_docs():
    documents = []
    for file in os.listdir(data):
        if file.endswith(".pdf"):
            file_path = os.path.join(data,file)
            loader = PyPDFLoader(file_path)
            pages = loader.load()

            for page in pages:
                page.metadata["source"] = file

            documents.extend(pages) 

    return documents

def clean_docs(documents):
    cleaned_docs = []
    for doc in documents:
        text = " ".join(doc.page_content.split())
        
        if "references" in text.lower():
            continue

        doc.page_content = text
        cleaned_docs.append(doc)

    return cleaned_docs


def chunk_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 120,
    )
    return splitter.split_documents(documents)

def build_vectorstore(chunks):
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding = OpenAIEmbeddings(),
        persist_directory=persist_dir,
    )
    vectordb.persist()


if __name__ == "__main__":
    print("Loading PDFs...")
    documents = load_docs()
    print(f"Loaded {len(documents)} pages")

    print("Cleaning documents...")
    documents = clean_docs(documents)

    print("Chunking documents...")
    chunks = chunk_docs(documents)
    print(f"Total chunks: {len(chunks)}")

    print("Creating embeddings and building vector store...")
    build_vectorstore(chunks)

    print("Ingestion completed successfully.")