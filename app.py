from flask import Flask, request, jsonify, Response,stream_with_context
# from llama_cpp import Llama
from mx_lm import load_model,create_text,create_text_stream
# ANSI escape code for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'
model, tokenizer=load_model()



app = Flask(__name__)



@app.route('/v1/g', methods=['POST'])
def v1_generate():
    data = request.json
    instruction = data.get('i', '')
    query = data.get('q', '')
    prompt = f"### Instruction:\n{instruction}\n### Query:\n{query}\n### Response:\n"
    response = create_text(tokenizer, model, prompt)
    return jsonify({'response': response})

@app.route('/v1/gs', methods=['POST'])
def v1_generate_stream():
    data = request.json
    instruction = data.get('i', '')
    query = data.get('q', '')
    prompt = f"### Instruction:\n{instruction}\n### Query:\n{query}\n### Response:\n"

    return Response(stream_with_context(create_text_stream(tokenizer, model, prompt)), mimetype="text/event-stream")

if __name__ == '__main__':
    app.run(debug=True, port=5005)


