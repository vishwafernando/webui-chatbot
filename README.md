# Hugging Face Text Generation WebUI (Gradio Edition)

A modern user interface for running Hugging Face text-generation models locally using Gradio.

## Features

- **Chat tab** — conversational UI with history, persona/impersonation, advanced controls
- **Notebook tab** — longform notes/prompt engineering scratchpad
- **Model selection** — choose from several Hugging Face (local) models
- **Advanced parameters** — temperature, top-k
- **Responsive UI** — responsive ui 
- **Runs fully local** — all generation happens on your machine

## Quickstart

1. Install Python 3.8+ and run:
    ```sh
    pip install gradio transformers torch huggingface_hub
    ```

2. To add a model (by link or ID):
    ```sh
    python add_model.py mistralai/Mistral-7B-Instruct-v0.2
    # or
    python add_model.py https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
    ```

    This will:
    - Download the model to `models/mistralai-Mistral-7B-Instruct-v0.2`
    - Auto-update `AVAILABLE_MODELS` in `webui_gradio.py` to include it

3. Run:
    ```sh
    python webui_gradio.py
    ```

4. Open [http://localhost:7860](http://localhost:7860) in your browser.

## Customizing

- Add your favorite Hugging Face models to the `models/` directory using `add_model.py`.
- Tweak the CSS in `webui_gradio.py` for a different look.
- For more options (max tokens, top-p, etc.), see the Gradio and Transformers docs.

## Credits

- Powered by [Gradio](https://gradio.app/), [Hugging Face Transformers](https://huggingface.co/docs/transformers/index), [huggingface_hub](https://huggingface.co/docs/huggingface_hub/index)
