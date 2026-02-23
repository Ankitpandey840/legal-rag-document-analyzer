Problem Statement

Legal contracts are long, complex, and difficult to analyze using traditional keyword-based search systems. Manual contract review is time-consuming, expensive, and prone to human error.

Organizations require automated systems that can:

Extract relevant clauses

Answer legal questions accurately

Maintain document traceability

Avoid hallucinated or fabricated information

Solution Approach

This system implements a complete RAG pipeline:

Parse PDF contracts

Perform semantic chunking with overlap

Generate dense embeddings using a transformer model

Store embeddings in a FAISS vector database

Retrieve top-K relevant legal clauses

Generate grounded answers using a deterministic LLM

The LLM is constrained to use only retrieved context to reduce hallucination risk.

System Architecture

PDF Documents
→ Text Parsing
→ Chunking
→ Embedding Generation
→ FAISS Vector Index
→ Semantic Retrieval
→ LLM Generation
→ Final Answer

Project Structure

legal-rag-document-analyzer

data
src
pdf_parser.py
chunking.py
embeddings.py
vector_store.py
retrieval.py
rag_generator.py

main.py
requirements.txt
README.md
.gitignore

Technologies Used

Python
Sentence Transformers
FAISS
HuggingFace Transformers
PyTorch
PDFPlumber

Installation

Step 1: Clone the repository

git clone https://github.com/YOUR_USERNAME/legal-rag-document-analyzer.git

Step 2: Navigate to project folder

cd legal-rag-document-analyzer

Step 3: Install dependencies

pip install -r requirements.txt

Usage

Add PDF files inside the data folder.

Run the application:

python main.py

Enter a legal question when prompted.

Key Features

Semantic search over legal documents
15k+ embedding support
FAISS similarity indexing
Deterministic LLM decoding to reduce hallucinations
Modular and production-style architecture
Page-level metadata traceability

Use Cases

Enterprise legal teams
Compliance departments
Contract risk analysis
Corporate governance automation

Author

Ankit Pandey
M.Tech Artificial Intelligence IIT PATNA