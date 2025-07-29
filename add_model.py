import sys
import os
import re
from huggingface_hub import snapshot_download

WEBUI_FILE = "webui_gradio.py"
MODELS_DIR = "models"

def model_id_from_link(link):
    # Accepts full URLs or IDs
    if link.startswith("https://huggingface.co/"):
        parts = link.replace("https://huggingface.co/", "").strip("/").split("/")
        # Remove "tree/main" or similar suffixes
        if "tree" in parts:
            parts = parts[:parts.index("tree")]
        return "/".join(parts)
    return link

def download_model(model_id):
    model_dir = os.path.join(MODELS_DIR, model_id.replace("/", "-"))
    print(f"Downloading '{model_id}' to '{model_dir}'...")
    snapshot_download(repo_id=model_id, local_dir=model_dir, local_dir_use_symlinks=False)
    print("Download complete.")
    return model_dir

def update_available_models(new_model_dir):
    with open(WEBUI_FILE, "r") as f:
        contents = f.read()
    # Find the AVAILABLE_MODELS list
    pattern = re.compile(r"(AVAILABLE_MODELS\s*=\s*\[\s*)([^]]*)(\])", re.MULTILINE)
    match = pattern.search(contents)
    if not match:
        raise RuntimeError("Could not find AVAILABLE_MODELS list in webui_gradio.py")

    # Extract current entries
    entries = [e.strip().strip('"').strip("'") for e in match.group(2).split(",") if e.strip()]
    # Add new entry if not present
    if new_model_dir not in entries:
        entries.append(new_model_dir)
    # Remove duplicates and sort
    entries = sorted(set(entries))
    # Create new list string
    new_list = ',\n    '.join([f'"{e}"' for e in entries])
    new_contents = pattern.sub(f"AVAILABLE_MODELS = [\n    {new_list}\n]", contents)

    with open(WEBUI_FILE, "w") as f:
        f.write(new_contents)
    print(f"Updated AVAILABLE_MODELS in {WEBUI_FILE}.")

def main():
    if len(sys.argv) != 2:
        print("Usage: python add_model.py <huggingface_model_link_or_id>")
        sys.exit(1)
    model_link = sys.argv[1]
    model_id = model_id_from_link(model_link)
    model_dir = os.path.join(MODELS_DIR, model_id.replace("/", "-"))
    download_model(model_id)
    update_available_models(model_dir)

if __name__ == "__main__":
    main()