import runpod
from llama_initializer import initialize_runner
from llama_client import send_completion_request

# Initialize the runner globally when the script is loaded
initialize_runner()

def handler(event):
    """
    This function processes incoming requests to your Serverless endpoint.

    Args:
        event (dict): Contains the input data and request metadata

    Returns:
        dict: The result to be returned to the client
    """
    print("Processing request...")

    # Extract input data
    input_data = event.get('input', {})
    prompt = input_data.get('prompt')

    if not prompt:
        return {"error": "No 'prompt' found in job input."}

    # Extract optional parameters
    max_tokens = input_data.get('max_tokens', 500)
    temperature = input_data.get('temperature', 0.7)

    # Send request to llama.cpp server
    result = send_completion_request(prompt, max_tokens, temperature)

    return result


runpod.serverless.start({'handler': handler})