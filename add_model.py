import sys
import os
import getpass
import re
from pathlib import Path
import subprocess

WEBUI_FILE = "webui_gradio.py"
MODELS_BASE = "user_data/models"

def run_download(model_id, threads=8):
    # Use your fast downloader with more threads for speed
    cmd = [
        sys.executable, "download-model.py",
        model_id,
        "--threads", str(threads)
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Download failed!")
        sys.exit(1)

def model_local_dir(model_id):
    username = getpass.getuser()
    safe_model_id = re.sub(r'[^a-zA-Z0-9_.-]', '_', model_id)
    return f"{MODELS_BASE}/{username}_{safe_model_id}"

def find_downloaded_folder(model_id):
    username = getpass.getuser()
    safe_model_id = re.sub(r'[^a-zA-Z0-9_.-]', '_', model_id)
    folder = f"{MODELS_BASE}/{username}_{safe_model_id}"
    if Path(folder).exists():
        return folder
    # Fallback: try to find a matching subfolder 
    for d in Path(MODELS_BASE).glob(f"{username}_*"):
        if safe_model_id in d.name:
            return str(d)
    return folder

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
    # Accept both full links or just IDs
    if model_link.startswith("https://huggingface.co/"):
        model_id = model_link.replace("https://huggingface.co/", "").strip("/")
    else:
        model_id = model_link
    run_download(model_id)
    local_dir = find_downloaded_folder(model_id)
    update_available_models(local_dir)

if __name__ == "__main__":
    main()
