from flask import Flask, request, jsonify, Response
from llama_cpp import Llama
import logging
from typing import Generator

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Global model instance
llm = None

def load_model():
    global llm
    try:
        model_path = "/Users/ashutoshpanchal/.lmstudio/models/chiizaraa/Llama-3.2-1B-Instruct-Q4_K_M-GGUF/llama-3.2-1b-instruct-q4_k_m.gguf"
        
        app.logger.info(f"Loading model from {model_path}...")
        
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,          # Context window size
            n_gpu_layers=-1,     # -1 = all layers to GPU
            n_threads=8,         # CPU threads
            verbose=False
        )
        
        app.logger.info("Model loaded successfully")
        
    except Exception as e:
        app.logger.error(f"Model loading failed: {str(e)}")
        raise
load_model()


@app.route('/g', methods=['POST'])
def generate_text():
    if not llm:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        instruction = data.get('i', '')
        query = data.get('q', '')
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
    
    except Exception as e:
        app.logger.error(f"Generation error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_stream', methods=['POST'])
def generate_stream():
    if not llm:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
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
    
    except Exception as e:
        app.logger.error(f"Streaming error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    if not llm:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        return jsonify({
            "model_name": llm.model_path,
            "context_size": llm.n_ctx(),
            "gpu_layers": llm.n_gpu_layers,
            "model_size": f"{llm.model.params.model_size / 1e9:.1f}B",
            "quantization": "Q4_K_M"  # Update based on your model file
        })
    
    except Exception as e:
        app.logger.error(f"Model info error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, threaded=True)