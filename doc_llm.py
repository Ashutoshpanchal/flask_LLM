import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from mlx_lm import load, generate

# === Step 1: Load and Clean Data ===
def load_txt_data(file_path):
    """
    Load and clean text data from a .txt file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    cleaned_data = [line.strip() for line in lines if line.strip()]
    return cleaned_data

def split_text_into_chunks(data, max_length=100):
    """
    Split each text entry into smaller chunks to improve embeddings.
    """
    chunks = []
    for entry in data:
        sentences = entry.split('. ')
        for sentence in sentences:
            if len(sentence) > max_length:
                chunks.extend([sentence[i:i + max_length] for i in range(0, len(sentence), max_length)])
            else:
                chunks.append(sentence)
    return chunks

# === Step 2: Initialize FAISS and Embedding Model ===
def initialize_faiss(vector_dim):
    """
    Initialize a FAISS index for storing vectors.
    """
    index = faiss.IndexFlatL2(vector_dim)
    return index

def generate_embeddings(data, embedding_model):
    """
    Generate embeddings for text data.
    """
    return embedding_model.encode(data, convert_to_tensor=False)

def store_embeddings_in_faiss(data, embeddings, index):
    """
    Store text and embeddings in FAISS.
    """
    index.add(np.array(embeddings, dtype=np.float32))
    return index

# === Step 3: Query FAISS and Retrieve Context ===
def query_faiss(index, embedding_model, query, data, k=3):
    """
    Query FAISS for the top k most similar results to the query.
    """
    query_emb = embedding_model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_emb, dtype=np.float32), k)
    results = [data[i] for i in indices[0] if i < len(data)]
    return results

# === Step 4: Load Llama Model and Generate Response ===
def load_model():
    """
    Load the Llama model and tokenizer.
    """
    model, tokenizer = load("/Users/ashutoshpanchal/.lmstudio/models/mlx-community/Llama-3.2-3B-Instruct-4bit/")
    return model, tokenizer

def create_prompt(context, query):
    """
    Create a structured prompt for the Llama model.
    """
    prompt = (
        f"Context:\n{context}\n\n"
        f"User Query:\n{query}\n\n"
        "Answer based on the context above:"
    )
    return prompt

def create_text(tokenizer, model, prompt):
    """
    Generate a response from the model.
    """
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    text = generate(model, tokenizer, prompt=prompt, verbose=True)
    return text

# === Step 5: Full Workflow ===
if __name__ == "__main__":
    # Path to your data file
    file_path = "/Users/ashutoshpanchal/Desktop/Project/project/flask_LLM/snapview.txt"  # Replace with the correct .txt file path

    # Load and clean data
    print("Loading and cleaning data...")
    raw_data = load_txt_data(file_path)
    cleaned_data = split_text_into_chunks(raw_data)

    # Initialize embedding model
    print("Generating embeddings...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = generate_embeddings(cleaned_data, embedding_model)

    # Initialize FAISS and store embeddings
    vector_dim = embeddings[0].shape[0]
    faiss_index = initialize_faiss(vector_dim)
    faiss_index = store_embeddings_in_faiss(cleaned_data, embeddings, faiss_index)
    print("Embeddings stored in FAISS!")

    # Load Llama model
    model, tokenizer = load_model()

    # Simulate user query
    user_query = "How do I change the period?"  # Replace with the user's query

    # Retrieve relevant context from FAISS
    print("Retrieving relevant context...")
    retrieved_context = query_faiss(faiss_index, embedding_model, user_query, cleaned_data, k=3)
    context = " ".join(retrieved_context)  # Combine retrieved chunks

    # Generate response from the model
    print("Generating response from the model...")
    prompt = create_prompt(context, user_query)
    response = create_text(tokenizer, model, prompt)

    # Output the response
    print("\n==========")
    print("Model Response:")
    print(response)
