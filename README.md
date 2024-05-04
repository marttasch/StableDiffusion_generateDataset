

## Requirements
- AUTOMATIC1111/stable-diffusion-webui with enabled API
    - [github](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- Automatic1111 Addons
    - [sd-webui-segmentation-anything](https://github.com/continue-revolution/sd-webui-segment-anything)
    
## Installation
- `pip install -r requirements.txt`
- check CUDA version with `nvcc --version`
   - or use NVIDIA-Control-Panel -> Help -> System Information -> Components -> CUDA Driver Version
- install torch and torchvision with the correct CUDA version
   - CUDA 11.8: `pip3 install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118`
   - CUDA 12.8: `pip3 install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121`


- `cp secrets.yml.example secrets.yml`
   - Fill in the secrets.yml with your API key and the URL of the API

