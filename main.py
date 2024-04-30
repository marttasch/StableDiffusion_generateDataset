import os
import random
import time

import src.A1111_api.A1111_api as sdwebui
import src.genPrompt.genPrompt as genPrompt

import logging

# ===== Conig =====
datasetName = 'urinal_v2'
maxGenerationCount = 10000

prompt = "puppy dog"   # default prompt, will be overwritten by genPrompt
negative_prompt = "(semi-realistic,cgi,3d,render,sketch,cartoon,drawing,anime),text,cropped,out of frame,cut off,(worst quality,low quality),jpeg artifacts,duplicate,(deformed),blurry,bad proportions,faucet,UnrealisticDream"
seed = -1   # will be overwritten
steps = 45
width = 512
height = 512
cfg_scale = 7
sampler_name = "DPM++ 2M"
n_iter = 1
batch_size = 1

payloadTxt2img = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "steps": steps,
        "width": width,
        "height": height,
        "cfg_scale": cfg_scale,
        "sampler_name": sampler_name,
        "n_iter": n_iter,
        "batch_size": batch_size,
}
payloadImg2img = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "steps": steps,
        "width": width,
        "height": height,
        "cfg_scale": cfg_scale,
        "sampler_name": sampler_name,
        "n_iter": n_iter,
        "batch_size": batch_size,
        "denoising_strength": 0.5,
        }
# ==== END Config ====

outputFolder = 'image_generation_output'
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
if not os.path.exists(os.path.join(outputFolder, datasetName)):
    os.makedirs(os.path.join(outputFolder, datasetName))
if not os.path.exists(os.path.join(outputFolder, datasetName, 'clean')):
    os.makedirs(os.path.join(outputFolder, datasetName, 'clean'))
if not os.path.exists(os.path.join(outputFolder, datasetName, 'avgDirty')):
    os.makedirs(os.path.join(outputFolder, datasetName, 'avgDirty'))
if not os.path.exists(os.path.join(outputFolder, datasetName, 'dirty')):
    os.makedirs(os.path.join(outputFolder, datasetName, 'dirty'))

loggingPath = os.path.join('api_out', datasetName, 'log.txt')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[
                        logging.FileHandler(loggingPath),
                        logging.StreamHandler()
                    ])

if __name__ == '__main__':
    startTime = time.time()

    api = sdwebui.APIHandler()
    logging.info('APIHandler initialized')

    try:
        # ==== Generate prompts ====
        logging.info('Generating prompts...')
        promptSets = genPrompt.genPrompts(configFile='./genPrompt/config.json')
        totalPrompts = promptSets['generic']
        totalImages = len(totalPrompts) * 3
        logging.info('total prompts: %s', len(promptSets['generic']))

        # save promtSets as json
        with open(os.path.join('api_out', datasetName, 'prompts.json'), 'w') as f:
            import json
            json.dump(promptSets, f, indent=4)
        with open(os.path.join('api_out', datasetName, 'prompts.txt'), 'w') as f:
            for prompt in totalPrompts:
                f.write(prompt['prompt'] + '\n')
        logging.info('Prompts written to %s', os.path.join('api_out', datasetName, 'prompts.json'))
        logging.info('Prompts written to %s', os.path.join('api_out', datasetName, 'prompts.txt'))

        # ==== Generate images ====
        promptCount = 0
        imageGeneratedCount = [0, 0, 0]
        set = promptSets['generic']
        #print('Prompt set:', set)
        for prompt in set:
            seed = random.randint(1000, 9999999999)
            # === CLEAN ===
            # generate clean image, txt2img
            payloadTxt2img['prompt'] = prompt['prompt'] + ', (clean)'
            payloadTxt2img['seed'] = seed

            # generate
            print('')
            logging.info(f'[{promptCount}/{totalImages}] Generating clean image...')
            logging.info('Prompt: %s', payloadTxt2img['prompt'])
            logging.info('Seed: %s', seed)
            out_dir = os.path.join('api_out', datasetName, 'clean')
            txt2imgPath, txt2imgResponse = api.call_txt2img_api(out_dir=out_dir , **payloadTxt2img)
            logging.info('Image saved to %s', txt2imgPath)
            imageGeneratedCount[0] += 1

            # === avgDirty ===
            # make it slightly dirty, img2img
            promptImg2img = prompt['prompt'] + ', (stains), <lora:dirtyStyle_LoRA_v2-000008:0.35>'
            payloadImg2img['prompt'] = promptImg2img
            payloadImg2img['seed'] = seed

            # load image as init_images
            init_images = [
                api.encode_file_to_base64(txt2imgPath),
            ]
            payloadImg2img['init_images'] = init_images

            # generate
            print('')
            logging.info(f'[{promptCount+1}/{totalImages}] Generating slightly dirty image...')
            logging.info('Prompt: %s', payloadImg2img['prompt'])
            logging.info('Seed: %s', seed)
            out_dir = os.path.join('api_out', datasetName, 'avgDirty')
            img2imgPath = api.call_img2img_api(out_dir=out_dir ,**payloadImg2img)
            logging.info('Image saved to %s', img2imgPath)
            imageGeneratedCount[1] += 1

            # === dirty ===
            # generate dirty image, img2img
            promptImg2img = prompt['prompt'] + ', (dirty, stains), <lora:dirtyStyle_LoRA_v2-000008:0.5>'
            payloadImg2img['prompt'] = promptImg2img
            payloadImg2img['seed'] = seed

            # generate
            print('')
            logging.info(f'[{promptCount+2}/{totalImages}] Generating dirty image...')
            logging.info('Prompt: %s', payloadImg2img['prompt'])
            logging.info('Seed: %s', seed)
            out_dir = os.path.join('api_out', datasetName, 'dirty')
            img2imgPath = api.call_img2img_api(out_dir=out_dir ,**payloadImg2img)
            logging.info('Image saved to %s', img2imgPath)
            imageGeneratedCount[2] += 1

            promptCount += 1
            if promptCount >= maxGenerationCount:
                break

        endTime = time.time()
        timeElapsed = endTime - startTime

        logging.info('== Generation complete! ==')
        logging.info('Prompt count: %s', promptCount)
        logging.info('Images generated: %s', imageGeneratedCount)
        logging.info('Total images: %s', totalImages)
        logging.info('Time elapsed: %s', timeElapsed)

    except Exception as e:
        logging.error('Error: %s', e)
        logging.error('Error: %s', str(e))
        logging.error('Error: %s', repr(e))
        logging.error('Error: %s', e.args)
        logging.error('Error: %s', e.with_traceback)
        logging.error('Error: %s', e.with_traceback())
        
        endTime = time.time()
        timeElapsed = endTime - startTime
        logging.info('Exiting...')
        logging.info('Prompt count: %s', promptCount)
        logging.info('Images generated: %s', imageGeneratedCount)
        logging.info('Total images: %s', totalImages)
        logging.info('Time elapsed: %s', timeElapsed)

    except KeyboardInterrupt as e:
        logging.error('KeyboardInterrupt: %s', e)

        endTime = time.time()
        timeElapsed = endTime - startTime
        logging.info('Exiting...')
        logging.info('Prompt count: %s', promptCount)
        logging.info('Images generated: %s', imageGeneratedCount)
        logging.info('Total images: %s', totalImages)
        logging.info('Time elapsed: %s', timeElapsed)