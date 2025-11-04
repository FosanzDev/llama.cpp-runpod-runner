import os
import subprocess
import json
import time
import requests
import sys

# Global variables
llama_server_process = None
is_server_ready = False
server_port = os.environ.get("SERVER_PORT", "9095")
server_host = "0.0.0.0"  # Fixed to 0.0.0.0 for RunPod compatibility
server_url = f"http://localhost:{server_port}"  # Use localhost for internal requests
hf_model_file_name = None # To store the model file name globally for handler
api_token = os.environ.get("API_TOKEN")  # Optional API token for authentication

def _execute_command(command, cwd=None):
    """Helper to execute shell commands and print output."""
    print(f"Executing command: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, cwd=cwd)
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

def initialize_runner():
    global llama_server_process, is_server_ready, server_port, server_url, hf_model_file_name, api_token

    print("Initializing Llama.cpp runner...")

    hf_token = os.environ.get("HUGGING_FACE_TOKEN")
    hf_model_repo = os.environ.get("HF_MODEL_REPO")
    hf_model_file_name = os.environ.get("HF_MODEL_FILE")
    server_config_file = os.environ.get("SERVER_CONFIG_FILE", "server_config.json")

    if not hf_token:
        print("ERROR: HUGGING_FACE_TOKEN environment variable not set.")
        sys.exit(1)
    if not hf_model_repo:
        print("ERROR: HF_MODEL_REPO environment variable not set.")
        sys.exit(1)
    if not hf_model_file_name:
        print("ERROR: HF_MODEL_FILE environment variable not set.")
        sys.exit(1)

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Installation steps
    try:
        _execute_command("pip install llama-cpp-python[server] --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124")
        _execute_command("pip install -U 'huggingface_hub'")
        _execute_command(f"hf auth login --token {hf_token}")
        _execute_command(f"hf download {hf_model_repo} {hf_model_file_name} --local-dir models/")
    except Exception as e:
        print(f"Failed during installation or model download: {e}")
        sys.exit(1)

    # Model parameters for server_config.json
    n_gpu_layers = int(os.environ.get("N_GPU_LAYERS", "-1"))
    n_ctx = int(os.environ.get("N_CTX", "40000"))
    n_batch = int(os.environ.get("N_BATCH", "512"))
    n_threads = int(os.environ.get("N_THREADS", "8"))
    offload_kqv = os.environ.get("OFFLOAD_KQV", "true").lower() == "true"
    use_mlock = os.environ.get("USE_MLOCK", "true").lower() == "true"
    rope_freq_scale = float(os.environ.get("ROPE_FREQ_SCALE", "4.0"))
    chat_format = os.environ.get("CHAT_FORMAT", "gemma")

    # Generate server_config.json
    config_content = {
        "host": server_host,
        "port": int(server_port),
        "models": [
            {
                "model": f"models/{hf_model_file_name}",
                "n_gpu_layers": n_gpu_layers,
                "n_ctx": n_ctx,
                "n_batch": n_batch,
                "n_threads": n_threads,
                "offload_kqv": offload_kqv,
                "use_mlock": use_mlock,
                "rope_freq_scale": rope_freq_scale,
                "chat_format": chat_format
            }
        ]
    }

    # Add API key if provided
    if api_token:
        config_content["api_key"] = api_token

    try:
        with open(server_config_file, "w") as f:
            json.dump(config_content, f, indent=2)
        print(f"Generated {server_config_file} with content:\n{json.dumps(config_content, indent=2)}")
    except IOError as e:
        print(f"ERROR: Failed to write server_config.json: {e}")
        sys.exit(1)

    # Start the llama_cpp.server in the background
    try:
        cmd = f"python3 -m llama_cpp.server --config_file {server_config_file}"
        print(f"Starting llama_cpp.server in background: {cmd}")
        llama_server_process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        print(f"llama_cpp.server process started with PID: {llama_server_process.pid}")
    except Exception as e:
        print(f"ERROR: Failed to start llama_cpp.server: {e}")
        sys.exit(1)

    # Wait for the server to become ready (10 minutes max)
    print(f"Waiting for llama_cpp.server to become ready at {server_url}...")
    for i in range(600):  # Try for up to 10 minutes (600 seconds)
        try:
            # Check the /health endpoint
            headers = {}
            if api_token:
                headers["Authorization"] = f"Bearer {api_token}"

            response = requests.get(f"{server_url}/health", timeout=5, headers=headers)
            if response.status_code == 200:
                print("Llama.cpp server is ready!")
                is_server_ready = True
                break
        except requests.exceptions.ConnectionError:
            pass # Server not up yet, ignore
        except Exception as e:
            print(f"Health check failed with unexpected error: {e}")

        if i % 10 == 0:  # Print status every 10 seconds
            print(f"Server not ready yet, retrying... ({i+1}/600) - {(i+1)/60:.1f} minutes elapsed")
        time.sleep(1)

    if not is_server_ready:
        print("ERROR: Llama.cpp server did not become ready in 10 minutes.")
        if llama_server_process:
            llama_server_process.kill() # Terminate the server process if it failed to start
        sys.exit(1)

def get_server_status():
    """Return current server status and configuration"""
    return {
        "is_ready": is_server_ready,
        "server_url": server_url,
        "model_file": hf_model_file_name,
        "api_token": api_token is not None
    }