from mlx_lm import load, generate,stream_generate



def load_model():
    model, tokenizer = load("/Users/ashutoshpanchal/.lmstudio/models/mlx-community/Llama-3.2-3B-Instruct-4bit/")
    return model, tokenizer

def create_text(tokenizer, model, prompt):
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )
    text = generate(model, tokenizer, prompt=prompt, verbose=True)
    return text

def create_text_stream(tokenizer, model, prompt):
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )
    for chuck in stream_generate(model, tokenizer, prompt=prompt):
        print("Data",chuck.text)
        yield chuck.text