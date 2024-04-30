import requests
import base64

# Define the URL and the payload to send.
url = "http://127.0.0.1:7860"

#api_endpoint = "sdapi/v1/txt2img"
#api_endpoint = "sdapi/v1/img2img"
#api_endpoint = "/sam/dino-predict"
api_endpoint = "/sam/sam-predict"

payload = {
    "prompt": "puppy dog",
    "steps": 5
}

payload_sam = {
  "sam_model_name": "sam_vit_h_4b8939.pth",
  "input_image": "D:/code/Bachelorarbeit/testSegmentation/testImg.png",
  "dino_enabled": True,
  "dino_model_name": "GroundingDINO_SwinT_OGC (694MB)",
  "dino_text_prompt": "toilet",
  "dino_box_threshold": 0.3,
  "dino_preview_checkbox": False,
  "dino_preview_boxes_selection": [
    0
  ]
}

payload_dino = {
  "input_image": "D:/code/Bachelorarbeit/testSegmentation/testImg.png",
  #"dino_model_name": "GroundingDINO_SwinT_OGC (694MB)",
  "dino_model_name": "GroundingDINO_SwinB (938MB)",
  "text_prompt": "toilet",
  "box_threshold": 0.3
}

# encode image 
payload_dino['input_image'] = base64.b64encode(open(payload_sam['input_image'], 'rb').read()).decode('utf-8')

# Send said payload to said URL through the API.
payload = payload_sam
response = requests.post(url=f'{url}{api_endpoint}', json=payload)
r = response.json()
print(r.keys())
print(len(r['masks']))
print(len(r['masked_images']))
print("Message:", r['msg'])
#print(r['image_with_box'])
#print(r)

# # Decode and save the image.
with open("output.png", 'wb') as f:
    #f.write(base64.b64decode(r['images'][0]))   # txt2img
    #f.write(base64.b64decode(r['image_with_box']))  # dino
    pass

# sam
for i in range(len(r['blended_images'])):
    with open(f"blendedImage_{i}.png", 'wb') as f:
        f.write(base64.b64decode(r['blended_images'][i]))

for i in range(len(r['masks'])):
    with open(f"mask_{i}.png", 'wb') as f:
        f.write(base64.b64decode(r['masks'][i]))

for i in range(len(r['masked_images'])):
    with open(f"maskedImage_{i}.png", 'wb') as f:
        f.write(base64.b64decode(r['masked_images'][i]))



