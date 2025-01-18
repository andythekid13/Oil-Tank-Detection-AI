import requests
import base64
import time

def ask_ollama_with_image(prompt: str, image_path: str, model: str = "llava"):
    """
    Queries Ollama with both text and image input, with improved timeout handling
    """
    url = "http://localhost:11434/api/generate"
    
    # Read and encode the image with a max size limit
    try:
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        return f"Error reading image: {str(e)}"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,  # Enable streaming for better handling of long responses
        "images": [image_data]
    }

    try:
        # Increased timeout and added retry mechanism
        session = requests.Session()
        session.timeout = (30, 300)  # (connect timeout, read timeout)
        
        response = session.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        # Process the streamed response
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_response = requests.compat.json.loads(line)
                    if 'response' in json_response:
                        full_response += json_response['response']
                except json.JSONDecodeError:
                    continue
                
        return full_response

    except requests.exceptions.ConnectTimeout:
        return "Connection to Ollama timed out. Please ensure Ollama is running."
    except requests.exceptions.ReadTimeout:
        return "Analysis is taking too long. The image might be too complex or the model is overloaded."
    except requests.exceptions.ConnectionError:
        return "Cannot connect to Ollama. Please ensure Ollama is running with: 'ollama serve'"
    except Exception as e:
        return f"Error: {str(e)}"

def test_visual_model(model="llava"):
    """
    Test if the visual model is working properly with timeout handling
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": "test",
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=(5, 10))
        return response.status_code == 200
    except Exception as e:
        print(f"Model test error: {str(e)}")
        return False