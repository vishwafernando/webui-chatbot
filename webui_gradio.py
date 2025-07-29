import gradio as gr
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

AVAILABLE_MODELS = [
    "./models/gpt2"
]

PIPELINES = {}

def get_pipe(model_path):
    if model_path not in PIPELINES:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        PIPELINES[model_path] = pipe
    return PIPELINES[model_path]

def chat_fn(
    message, history, persona, model, temperature, top_k
):
    if persona.strip():
        prompt = f"[Persona: {persona.strip()}]\n{message}"
    else:
        prompt = message

    pipe = get_pipe(model)
    # To simulate chat, concat history into prompt if needed
    full_prompt = ""
    if history:
        for user, bot in history:
            full_prompt += f"User: {user}\nBot: {bot}\n"
        full_prompt += f"User: {prompt}\nBot:"
    else:
        full_prompt = prompt

    result = pipe(
        full_prompt,
        temperature=temperature,
        top_k=top_k,
        max_new_tokens=256,
        num_return_sequences=1,
        pad_token_id=pipe.tokenizer.eos_token_id
    )
    response = result[0]['generated_text']
    # Post-process to get only bot reply
    if response.startswith(full_prompt):
        response = response[len(full_prompt):].strip()
    return response

def chat_interface(
    message, history, persona, model, temperature, top_k
):
    response = chat_fn(message, history, persona, model, temperature, top_k)
    history = history + [[message, response]]
    return history, history

def regenerate_last(history, persona, model, temperature, top_k):
    if not history:
        return history, history
    # Remove last bot reply and regenerate
    last_user = history[-1][0]
    history = history[:-1]
    response = chat_fn(last_user, history, persona, model, temperature, top_k)
    history = history + [[last_user, response]]
    return history, history

def continue_last(history, persona, model, temperature, top_k):
    # Continue from last bot response
    if not history:
        return history, history
    last_bot = history[-1][1]
    # Use as user prompt
    response = chat_fn(last_bot, history, persona, model, temperature, top_k)
    history = history + [[last_bot, response]]
    return history, history

def remove_last(history):
    if not history:
        return history, history
    history = history[:-1]
    return history, history

with gr.Blocks(css="""
body, .gradio-container { background: #23272e !important; color: #f4f4f4; }
.gradio-container { font-family: Arial, sans-serif; }
#chatbox label, #notebookbox label { color: #fff; }
#chatbox textarea, #notebookbox textarea, .input-text, .input-number, select { background: #2e323a; color: #fff; border-radius: 4px; border: 1px solid #333;}
#chatbox .chat-history, #notebookbox textarea { background: #181c23; }
#chatbox .chat-history { min-height: 300px; max-height: 300px; overflow-y: auto; border-radius: 8px; padding: 10px; margin-bottom: 10px;}
""") as demo:
    gr.Markdown(
        """
        # Hugging Face Text Generation WebUI  
        <span style="color:#aaa">A Gradio-powered interface inspired by [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui)</span>
        """
    )
    with gr.Tab("Chat"):
        with gr.Row():
            persona = gr.Textbox(label="Persona (Impersonate)", placeholder="e.g. Shakespeare, Yoda, etc.", scale=2)
            model = gr.Dropdown(choices=AVAILABLE_MODELS, value=AVAILABLE_MODELS[0], label="Model", scale=2)
            temperature = gr.Slider(0, 2, value=1.0, label="Temperature", step=0.01, scale=1)
            top_k = gr.Slider(0, 100, value=40, label="Top-k", step=1, scale=1)
        with gr.Column(elem_id="chatbox"):
            chatbox = gr.Chatbot(label="Chat History", elem_classes=["chat-history"])
            msg = gr.Textbox(label="Your Message", placeholder="Type a prompt...", lines=2)
            with gr.Row():
                send_btn = gr.Button("Generate", variant="primary")
                regen_btn = gr.Button("Regenerate")
                cont_btn = gr.Button("Continue")
                remove_btn = gr.Button("Remove Last")
        state = gr.State([])  # Chat history 

        send_btn.click(fn=chat_interface, 
            inputs=[msg, state, persona, model, temperature, top_k], 
            outputs=[chatbox, state], queue=True)
        regen_btn.click(fn=regenerate_last, 
            inputs=[state, persona, model, temperature, top_k], 
            outputs=[chatbox, state], queue=True)
        cont_btn.click(fn=continue_last, 
            inputs=[state, persona, model, temperature, top_k], 
            outputs=[chatbox, state], queue=True)
        remove_btn.click(fn=remove_last,
            inputs=[state], outputs=[chatbox, state])
        msg.submit(fn=chat_interface, 
            inputs=[msg, state, persona, model, temperature, top_k], 
            outputs=[chatbox, state], queue=True)
    with gr.Tab("Notebook"):
        with gr.Column(elem_id="notebookbox"):
            gr.Markdown("Longform writing, notes, or prompt engineering scratchpad.")
            notebook = gr.Textbox(label="Notebook", placeholder="Write notes...", lines=15)
    with gr.Tab("Default"):
        gr.Markdown("""
        ## Instructions  
        - Use the **Chat** tab for interactive conversations with your model, including persona/impersonation and chat history.<br>
        - The **Notebook** tab is for your own notes and experimentation.<br>
        - Select the model, temperature, and top-k as needed.<br>
        - All model inference is local, using Hugging Face Transformers.
        """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)