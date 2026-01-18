# HealthWise AI
HealthWise AI is a RAG chatbot built for the purpose of answering specific questions, in our case healthcare related questions. Instead of relying on general knowledge, it retrieves relevant information from curated research papers and returns the answer along with citations.

--
## Project Goals
The project objectives are:
 - building a chatbot that answers questions only using the uploaded documents
 - uses vector search to retrieve the relevant information
 - generates concise answers
 - shows citations(document name and page number)
 - does not answer when the question is not relevant to the documents
 - provides a simple UI for user interaction

 ## Chatbot Details
 Standard RAG pipeline:
- Document Ingestion - Healthcare research PDFs are loaded, cleaned, split into overlapping chunks, embedded, and stored in a Chroma vector database.
- Retrieval & Relevance- For each  question, the system performs vector similarity search to retrieve candidate chunks which are then reranked using an LLM to improve relevance before answering.
- Answer Generation & Grounding- The final answer is generated strictly from the retrieved context.If the system cannot confidently answer using the documents, it refuses with an “I don’t know” response.
- Safety & Guardrails - Prompt-injection detection is applied, and the model is restricted to document-only knowledge to prevent hallucinations.
- Conversation Handling - Session-based chat history is maintained in the UI, while retrieval remains stateless to preserve grounding.
- Observability - Each interaction is logged with latency, answer/refusal status, retrieval signals, visualized in a simple dashboard.


- Must-have Requirements
    Document ingestion pipeline
    Vector similarity search (Top-K retrieval)
    Grounded answer generation
    Source citations (document + page number)
    Safe “I don’t know” handling
    Streamlit-based chat UI

- Nice-to-have Features
    LLM-based reranking
    Prompt-injection guardrails
    Observability dashboard (logs + charts)

## Structure
├── app/
│ └── app.py # Streamlit UI
├── rag/
│ ├── ingestion.py # PDF ingestion and vector indexing
│ └── retriever.py # Retrieval, reranking, guardrails
├── data/
│ ├── raw/healthcare/ # Healthcare research PDFs
│ └── observability/ # Interaction logs
├── index/ #  Chroma vector database
├── requirements.txt
└── README.md

## Setup
- Clone the repository
bash
git clone <repo-url>
cd<repo>
- Activate virtual environment
python -m venv .venv
.venv\\Scripts\\activate
- Install dependencies
pip install -r requirements.txt
- Set environment variables - API key in the .env file
- Run the ingestion pipeline
python rag/ingestion.py
- Running the app locally
streamlit run app/app.py

## Deployment
The application is deployed using Streamlit Community Cloud. It automatically redeploys on every push to Github repository.
Demo URL: https://healthwiseai.streamlit.app/


