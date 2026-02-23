import os

from src.pdf_parser import parse_pdf_directory
from src.chunking import create_chunks
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.retrieval import Retriever
from src.rag_generator import RAGGenerator


def main():

    pdf_directory = "data"  # Add PDF files inside the data folder

    if not os.path.exists(pdf_directory):
        print("Error: 'data' folder not found. Please create it and add PDF files.")
        return

    print("Parsing PDF documents...")
    pages = parse_pdf_directory(pdf_directory)
    print(f"Total pages parsed: {len(pages)}")

    print("Creating text chunks...")
    chunks = create_chunks(pages)
    print(f"Total chunks created: {len(chunks)}")

    texts = [chunk["chunk_text"] for chunk in chunks]

    print("Generating embeddings...")
    embedding_model = EmbeddingModel()
    embeddings = embedding_model.encode_texts(texts)

    print("Building FAISS vector index...")
    vector_store = VectorStore(dimension=embeddings.shape[1])
    vector_store.add_embeddings(embeddings)

    print("Initializing retriever...")
    retriever = Retriever(embedding_model, vector_store, chunks)

    print("Loading RAG generator model...")
    rag_generator = RAGGenerator()

    while True:
        question = input("\nEnter your legal question (type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        retrieved_chunks = retriever.retrieve(question)
        answer = rag_generator.generate_answer(question, retrieved_chunks)

        print("\nGenerated Answer:")
        print(answer)


if __name__ == "__main__":
    main()