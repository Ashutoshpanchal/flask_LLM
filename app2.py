from flask import Flask, request, jsonify, Response
from llama_cpp import Llama
import logging
from typing import Generator
import numpy as np
import os
import faiss
from doc_llm import ( split_text_into_chunks, initialize_faiss,
                      generate_embeddings, store_embeddings_in_faiss)
from sentence_transformers import SentenceTransformer


app = Flask(__name__)
app.logger.setLevel(logging.INFO)
import json
# Global model instance
llm = None

INDEX_FILE_PATH = 'faiss_index.index'
CLEANED_DATA_FILE_PATH = 'cleaned_data.json'  # Store cleaned data in this file

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


INDEX_FILE_PATH = 'faiss_index.index'

# Function to create or load the FAISS index
def create_or_load_faiss_index(dimension):
    if os.path.exists(INDEX_FILE_PATH):
        # If the index file exists, load it
        index = faiss.read_index(INDEX_FILE_PATH)
        print(f"Loaded existing FAISS index from {INDEX_FILE_PATH}")
    else:
        # Otherwise, create a new index
        index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean distance)
        print(f"Created a new FAISS index")
    return index

# Function to save the FAISS index
def save_faiss_index(index,INDEX_FILE_PATH):
    faiss.write_index(index, INDEX_FILE_PATH)
    print(f"Saved FAISS index to {INDEX_FILE_PATH}")


def load_faiss_index(index_file_path, vector_dim):
    """
    Load the FAISS index from a file if it exists, or initialize a new index.
    """
    if os.path.exists(index_file_path):
        # If index exists, load it
        print(f"Loading FAISS index from {index_file_path}")
        return faiss.read_index(index_file_path)
    else:
        # If the index does not exist, create a new one
        print(f"FAISS index not found. Creating a new index.")
        return initialize_faiss(vector_dim)


def query_faiss(index, embedding_model, query, data, k=3):
    """
    Query FAISS for the top k most similar results to the query.
    """
    query_emb = embedding_model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_emb, dtype=np.float32), k)
    results = [data[i] for i in indices[0] if i < len(data)]
    return results


def load_model():
    global llm
    
    # model_path = "/home/magenta/.cache/huggingface/hub/models--chiizaraa--Llama-3.2-1B-Instruct-Q4_K_M-GGUF/snapshots/d6fc1de754cf625e9cefdf3da24767bd3c8717c4/llama-3.2-1b-instruct-q4_k_m.gguf"
    model_path="/Users/ashutoshpanchal/.lmstudio/models/chiizaraa/Llama-3.2-1B-Instruct-Q4_K_M-GGUF/llama-3.2-1b-instruct-q4_k_m.gguf"
    
    app.logger.info(f"Loading model from {model_path}...")
    
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,          # Context window size
        n_gpu_layers=-1,     # -1 = all layers to GPU
        n_threads=8,         # CPU threads
        verbose=False
    )
    
    app.logger.info("Model loaded successfully")

def load_cleaned_data(cleaned_data_file_path):
    if os.path.exists(cleaned_data_file_path):
        print("Loading cleaned data...",flush=True)
        with open(cleaned_data_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print("Cleaned data file not found. Returning empty list.",flush=True)
        return []

def load_text(file):
    content = file.read().decode('utf-8')  # Read and decode the content
    lines = content.splitlines()  # Split into lines
    cleaned_data = [line.strip() for line in lines if line.strip()]  # Clean and remove empty lines
    return cleaned_data
load_model()


def parse_input():
    if request.method == 'POST':
        # For POST requests, get JSON data from the request body
        data = request.get_json()
    else:
        # For GET requests, parse JSON from the 'data' query parameter
        input_data = request.args.get('data')
        if not input_data:
            raise ValueError("Missing 'data' parameter")
        data = json.loads(input_data)
    
    instruction = data.get('i', '')
    query = data.get('q', '')
    return instruction, query

def save_cleaned_data(cleaned_data, cleaned_data_file_path):
    with open(cleaned_data_file_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f)
    print(f"Cleaned data saved to {cleaned_data_file_path}")


@app.route('/g', methods=['POST','GET'])
def generate_text():
    if not llm:
        return jsonify({"error": "Model not loaded"}), 500
    
    # try:
    # data = request.get_json()
    # instruction = data.get('i', '')
    # query = data.get('q', '')
    instruction, query = parse_input()
    prompt = f"### Instruction:\n{instruction}\n### Query:\n{query}\n### Response:\n"
    
    messages = [{"role": "user", "content": prompt}]
    
    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=500,
        stop=["<|eot_id|>"]
    )
    
    return jsonify({
        "response": response['choices'][0]['message']['content']
    })


@app.route('/generate_stream', methods=['GET','POST'])
def generate_stream():

    
        # data = request.get_json()
        # instruction = data.get('i', '')
        # query = data.get('q', '')
        instruction, query = parse_input()
        prompt = f"### Instruction:\n{instruction}\n### Query:\n{query}\n### Response:\n"
        
        messages = [{"role": "user", "content": prompt}]

        def generator():
            stream = llm.create_chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                stream=True,
                stop=["<|eot_id|>"]
            )
            
            for chunk in stream:
                if 'content' in chunk['choices'][0]['delta']:
                    yield chunk['choices'][0]['delta']['content']

        return Response(generator(), mimetype='text/plain')




@app.route('/m/g', methods=['POST','GET'])
def m_g():
    if not llm:
        return jsonify({"error": "Model not loaded"}), 500
    
    instruction, user_query = parse_input()
    cleaned_data = load_cleaned_data(CLEANED_DATA_FILE_PATH)  # Load the previously stored cleaned data
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    faiss_index = load_faiss_index(INDEX_FILE_PATH, vector_dim=384)  # 384 is the output dimension of 'all-MiniLM-L6-v2'
    retrieved_context = query_faiss(faiss_index, embedding_model, user_query, cleaned_data, k=3)
    context = " ".join(retrieved_context)
    prompt=create_prompt(context, user_query)
    
    messages = [{"role": "user", "content": prompt}]
    
    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=500,
        stop=["<|eot_id|>"]
    )
    
    return jsonify({
        "response": response['choices'][0]['message']['content']
    })




@app.route('/', methods=['GET'])
def model_info():
    if not llm:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        return jsonify({
            "model_name": llm.model_path,
            "context_size": llm.n_ctx(),
            "model_size": f"{llm.model.params.model_size / 1e9:.1f}B",
            "quantization": "Q4_K_M"  # Update based on your model file
        })
    
    except Exception as e:
        app.logger.error(f"Model info error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload-txt', methods=['POST'])
def upload_txt():
        # Check if a file is provided in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        # Get the file from the request
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        raw_data = load_text(file)
        
        # Read the .txt file content
        if os.path.exists(CLEANED_DATA_FILE_PATH):
            os.remove(CLEANED_DATA_FILE_PATH)
        if os.path.exists(INDEX_FILE_PATH):
            os.remove(INDEX_FILE_PATH)
        print("FAISS index is empty, loading and processing data...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Load and clean data
        cleaned_data = split_text_into_chunks(raw_data)

        # Generate embeddings for the data
        embeddings = generate_embeddings(cleaned_data, embedding_model)
        faiss_index = load_faiss_index(INDEX_FILE_PATH, vector_dim=384)  # 384 is the output dimension of 'all-MiniLM-L6-v2'

        # Store embeddings in the FAISS index
        faiss_index = store_embeddings_in_faiss(cleaned_data, embeddings, faiss_index)
        save_faiss_index(faiss_index, INDEX_FILE_PATH)
        print("Embeddings stored in FAISS!")

        # Save cleaned data to a file
        save_cleaned_data(cleaned_data, CLEANED_DATA_FILE_PATH)

        return jsonify({"message": "File uploaded and FAISS index updated successfully."}), 200






if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, threaded=True,debug=True)