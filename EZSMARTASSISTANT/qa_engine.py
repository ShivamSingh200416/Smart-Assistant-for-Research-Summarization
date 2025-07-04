from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load a pre-trained question-answering model
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def chunk_text(text, chunk_size=500):
    sentences = text.split(". ")
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def answer_question(question, context_text):
    chunks = chunk_text(context_text)
    
    # Select top matching chunks using TF-IDF
    vectorizer = TfidfVectorizer().fit(chunks + [question])
    vectors = vectorizer.transform(chunks + [question])
    sim_scores = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    top_indices = sim_scores.argsort()[-3:][::-1]

    top_chunks = [chunks[i] for i in top_indices]

    # Search for the best answer from top chunks
    best_score, best_answer, justification = -1, "", ""
    for chunk in top_chunks:
        try:
            result = qa_pipeline(question=question, context=chunk)
            if result["score"] > best_score:
                best_score = result["score"]
                best_answer = result["answer"]
                justification = chunk
        except:
            continue

    return best_answer or "No confident answer found.", justification