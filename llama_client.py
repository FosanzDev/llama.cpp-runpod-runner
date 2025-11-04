import requests
import json
from llama_initializer import get_server_status


def send_completion_request(prompt, max_tokens=500, temperature=0.7):
    """
    Send a chat completion request to the llama.cpp server

    Args:
        prompt (str): The user prompt
        max_tokens (int): Maximum tokens to generate
        temperature (float): Temperature for generation

    Returns:
        dict: Response from the server or error information
    """
    server_status = get_server_status()

    if not server_status["is_ready"]:
        return {"error": "Llama.cpp server is not ready. Initialization failed."}

    if not prompt:
        return {"error": "No prompt provided."}

    # Prepare request payload for the llama_cpp.server (OpenAI compatible chat completions)
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": server_status["model_file"],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    headers = {"Content-Type": "application/json"}
    if server_status["api_token"]:
        headers["Authorization"] = f"Bearer {server_status['api_token']}"

    try:
        response = requests.post(
            f"{server_status['server_url']}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=120
        )
        response.raise_for_status()
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
        print(f"An unexpected error occurred in client: {e}")
        return {"error": f"An unexpected error occurred: {e}"}