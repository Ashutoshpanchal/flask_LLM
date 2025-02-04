from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load LLM model (e.g., Mistral, Llama2)
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
chat_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load Embedding model (e.g., all-MiniLM)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({"models": [MODEL_NAME]})


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.json
    messages = data.get("messages", [])
    prompt = "\n".join([msg["content"] for msg in messages])

    response = chat_pipeline(prompt, max_length=200, num_return_sequences=1)
    return jsonify({"choices": [{"message": {"role": "assistant", "content": response[0]['generated_text']}}]})


@app.route("/v1/completions", methods=["POST"])
def completions():
    data = request.json
    prompt = data.get("prompt", "")

    response = chat_pipeline(prompt, max_length=200, num_return_sequences=1)
    return jsonify({"choices": [{"text": response[0]['generated_text']}]})

  
@app.route("/v1/embeddings", methods=["POST"])
def embeddings():
    data = request.json
    input_texts = data.get("input", [])
    
    if isinstance(input_texts, str):
        input_texts = [input_texts]

    embeddings = embedding_model.encode(input_texts).tolist()
    return jsonify({"data": [{"embedding": emb} for emb in embeddings]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=134, debug=True)
