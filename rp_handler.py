import os
import subprocess
import runpod
import json
import time
import requests
import sys

# Global variables
llama_server_process = None
is_server_ready = False
server_port = os.environ.get("SERVER_PORT", "9095")
server_host = "0.0.0.0"  # Fixed to 0.0.0.0 for RunPod compatibility
server_url = f"http://localhost:{server_port}"  # Use localhost for internal requests (test)
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

def _initialize_runner():
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
    for i in range(300):  # Try for up to 5 minutes
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

# Initialize the runner globally when the script is loaded
_initialize_runner()

def handler(job):
    global is_server_ready, server_url, hf_model_file_name, api_token

    if not is_server_ready:
        return {"error": "Llama.cpp server is not ready. Initialization failed."}

    # Extract prompt from job input
    prompt = job["input"].get("prompt")
    if not prompt:
        return {"error": "No 'prompt' found in job input."}

    # Prepare request payload for the llama_cpp.server (OpenAI compatible chat completions)
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": hf_model_file_name, # Use the dynamic model file name as the model identifier
        "max_tokens": job["input"].get("max_tokens", 500),
        "temperature": job["input"].get("temperature", 0.7)
    }

    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    try:
        response = requests.post(f"{server_url}/v1/chat/completions", json=payload, headers=headers, timeout=120)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        result = response.json()

        # Extract the generated content from the OpenAI-compatible response
        if result and result.get("choices"):
            generated_text = result["choices"][0]["message"]["content"]
            return {"result": generated_text}
        else:
            return {"error": f"Unexpected response format from LLM server: {json.dumps(result)}"}

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with LLM server: {e}")
        # Attempt to get more details if it's an HTTP error with a response body
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                return {"error": f"Failed to get response from LLM server: {e}", "details": error_details}
            except json.JSONDecodeError:
                return {"error": f"Failed to get response from LLM server: {e}", "response_text": e.response.text}
        return {"error": f"Failed to get response from LLM server: {e}"}
    except Exception as e:
        print(f"An unexpected error occurred in handler: {e}")
        return {"error": f"An unexpected error occurred: {e}"}

# Start the RunPod serverless worker.
# This line is called after the global _initialize_runner() has completed.
runpod.serverless.start({"handler": handler})