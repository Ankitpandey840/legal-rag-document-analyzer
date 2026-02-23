from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class RAGGenerator:
    """
    Handles LLM-based answer generation using retrieved context.
    """

    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def build_prompt(self, question: str, retrieved_chunks: List[Dict]) -> str:
        """
        Construct a strict legal-safe RAG prompt.
        """
        context = "\n\n".join(
            [f"[Source {i+1}] {chunk['chunk_text']}" for i, chunk in enumerate(retrieved_chunks)]
        )

        prompt = f"""
You are a legal document analysis assistant.

RULES:
- Answer STRICTLY using the provided context.
- Do NOT use external knowledge.
- If answer is not present, say: NOT FOUND IN DOCUMENT.

Context:
{context}

Question:
{question}

Answer:
"""
        return prompt.strip()

    def generate_answer(self, question: str, retrieved_chunks: List[Dict]) -> str:
        """
        Generate final answer using LLM.
        """
        prompt = self.build_prompt(question, retrieved_chunks)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False,     # deterministic (legal-safe)
                num_beams=4,
                early_stopping=True
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()