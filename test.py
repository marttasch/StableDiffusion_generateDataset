import requests
import base64

# Define the URL and the payload to send.
url = "http://127.0.0.1:7860"

payload = {
    "prompt": "puppy dog",
    "steps": 5
}

# Send said payload to said URL through the API.
response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
r = response.json()

# Decode and save the image.
with open("output.png", 'wb') as f:
    f.write(base64.b64decode(r['images'][0]))
