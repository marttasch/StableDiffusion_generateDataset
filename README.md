# Stabel Diffusion - generate Dataset
This repository contains the code for the Experiment of my Bachelor Thesis. The goal is to use Stable Diffusion to generate a trainingsdataset for a image recognition model, like ResNET50 or InceptionV3.
This Project uses the [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) with enabled API to generate the images. The generated images can be segmented with the [sd-webui-segmentation-anything](https://github.com/continue-revolution/sd-webui-segment-anything) Plugin. The Segmented images can then be used to train a image recognition model. Also some Validation Options are available to validate the trained Model and evaluate the generated images.

## Usage
### general
- Prompt-Config: `prompt_config.json` contains a Prompt-Template wich will be used to generate a list of variations of prompts
- Image Generation: `generateDataset.py` uses the A1111 API to generate images and split them into train, test and validation sets
- CNN Training: `trainCNN.py` uses the generated images to train a CNN model
- CNN Evaluation: `validateCNN.py` uses the trained model to validate with the validation set or evaluate on custom Folder

The Scripts can be controlled by passing arguments or by using the Jupyter Notebook UI

### Jupyter Notebook UI
Simple UI Elements, to start the Scripts

`01_main.ipynb`
- Test Prompt-Templates and generate Prompt-List
- Start image generation process using a Prompt-Config
- Start CNN Training

`02_eval.ipynb`
- Evaluate a trained Model with the Validation Set
- Evaluate a trained Model with a custom Folder
- Use Model to predict a single Image

`03_plots.ipynb`
- was used to generate Plots for the Thesis


## Requirements
- AUTOMATIC1111/stable-diffusion-webui with enabled API
    - [Github](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
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
   - currently used for Gotify Notifications

