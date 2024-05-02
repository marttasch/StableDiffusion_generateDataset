from datetime import datetime
import requests
import base64
import time
import os

class APIHandler:
    def __init__(self):
        self.webui_server_url = 'http://127.0.0.1:7860'
        self.out_dir = 'api_out'
        self.out_dir_t2i = os.path.join(self.out_dir, 'txt2img')
        self.out_dir_i2i = os.path.join(self.out_dir, 'img2img')
        os.makedirs(self.out_dir_t2i, exist_ok=True)
        os.makedirs(self.out_dir_i2i, exist_ok=True)

    @staticmethod
    def timestamp():
        return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    @staticmethod
    def encode_file_to_base64(path):
        with open(path, 'rb') as file:
            return base64.b64encode(file.read()).decode('utf-8')

    @staticmethod
    def decode_and_save_base64(base64_str, save_path):
        with open(save_path, "wb") as file:
            file.write(base64.b64decode(base64_str))

    def call_api(self, api_endpoint, **payload):
        response = requests.post(f'{self.webui_server_url}/{api_endpoint}', json=payload)
        return response.json()

    def call_txt2img_api(self, fileName=None, out_dir=None, **payload):
        response = self.call_api('sdapi/v1/txt2img', **payload)

        if fileName is None:
            fileName = f'txt2img-{self.timestamp()}'
        
        for index, image in enumerate(response.get('images')):
            if out_dir:
                save_path = os.path.join(out_dir, (fileName + f'_{index}.png'))
            else:
                save_path = os.path.join(self.out_dir_t2i, (fileName + f'_{index}.png'))
            self.decode_and_save_base64(image, save_path)
        return save_path, response

    def call_img2img_api(self, fileName=None, out_dir=None, **payload):
        response = self.call_api('sdapi/v1/img2img', **payload)

        if fileName is None:
            fileName = f'img2img-{self.timestamp()}'

        for index, image in enumerate(response.get('images')):
            if out_dir:
                save_path = os.path.join(out_dir, (fileName + f'_{index}.png'))
            else:
                save_path = os.path.join(self.out_dir_i2i, (fileName + f'_{index}.png'))
            self.decode_and_save_base64(image, save_path)
        return save_path, response
    
    # sam
    def call_sam_predict_api(self, decoded_image, samModelName="sam_vit_h_4b8939.pth", dinoEnabled=False, dinoPrompt="", dino_model_name="GroundingDINO_SwinT_OGC (694MB)"):
        payload_sam = {
            "sam_model_name": samModelName,
            "input_image": decoded_image,
            "dino_enabled": dinoEnabled,
            "dino_model_name": dino_model_name,
            "dino_text_prompt": dinoPrompt,
            "dino_box_threshold": 0.3,
            "dino_preview_checkbox": False,
            }
        
        r = self.call_api('sam/sam-predict', **payload_sam)

        msg = r['msg']
        blended_images = r['blended_images']
        masks = r['masks']
        masked_images = r['masked_images']

        return msg, blended_images, masks, masked_images

    
if __name__ == '__main__':
    payload = {
        "prompt": "masterpiece, (best quality:1.1), puppy",  # extra networks also in prompts
        "negative_prompt": "",
        "seed": 1,
        "steps": 20,
        "width": 512,
        "height": 512,
        "cfg_scale": 7,
        "sampler_name": "DPM++ 2M Karras",
        "n_iter": 1,
        "batch_size": 1,
    }

    api = APIHandler()
    api.call_img2img_api(**payload)

    # init_images = [
    #     encode_file_to_base64(r"B:\path\to\img_1.png"),
    #     # encode_file_to_base64(r"B:\path\to\img_2.png"),
    #     # "https://image.can/also/be/a/http/url.png",
    # ]

    # batch_size = 2
    # payload = {
    #     "prompt": "1girl, blue hair",
    #     "seed": 1,
    #     "steps": 20,
    #     "width": 512,
    #     "height": 512,
    #     "denoising_strength": 0.5,
    #     "n_iter": 1,
    #     "init_images": init_images,
    #     "batch_size": batch_size if len(init_images) == 1 else len(init_images),
    #     # "mask": encode_file_to_base64(r"B:\path\to\mask.png")
    # }
    # # if len(init_images) > 1 then batch_size should be == len(init_images)
    # # else if len(init_images) == 1 then batch_size can be any value int >= 1
    # call_img2img_api(**payload)

    # # there exist a useful extension that allows converting of webui calls to api payload
    # # particularly useful when you wish setup arguments of extensions and scripts
    # # https://github.com/huchenlei/sd-webui-api-payload-display
