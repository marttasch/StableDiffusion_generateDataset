import os
import random
import time

import src.A1111_api.A1111_api as sdwebui
import src.genPrompt.genPrompt as genPrompt

import logging

# ===== Conig =====
datasetName = 'urinal_v3-testing'
maxGenerationCount = 10000

prompt = "puppy dog"   # default prompt, will be overwritten by genPrompt
negative_prompt = "(semi-realistic,cgi,3d,render,sketch,cartoon,drawing,anime),text,cropped,out of frame,cut off,(worst quality,low quality),jpeg artifacts,duplicate,(deformed),blurry,bad proportions,faucet,UnrealisticDream"
seed = -1   # will be overwritten
steps = 20
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

# Create output folders
outputFolder = 'image_generation_output'
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
if not os.path.exists(os.path.join(outputFolder, datasetName)):
    os.makedirs(os.path.join(outputFolder, datasetName))
else:
    # print Output folder already exists. Overwrite??
    userInput = input(f'WARNING: Output folder {datasetName} already exists. Overwrite Files in this folder? (y/n): ')
    if userInput.lower() != 'y':
        print('Exiting...')
        exit()
if not os.path.exists(os.path.join(outputFolder, datasetName, 'clean')):
    os.makedirs(os.path.join(outputFolder, datasetName, 'clean'))
if not os.path.exists(os.path.join(outputFolder, datasetName, 'avgDirty')):
    os.makedirs(os.path.join(outputFolder, datasetName, 'avgDirty'))
if not os.path.exists(os.path.join(outputFolder, datasetName, 'dirty')):
    os.makedirs(os.path.join(outputFolder, datasetName, 'dirty'))

# Logging
loggingPath = os.path.join(outputFolder, datasetName, 'log.txt')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[
                        logging.FileHandler(loggingPath, mode='w'),
                        logging.StreamHandler()
                    ])

# Variables
promptCount = 1
imageCount = 1
imageGeneratedCount = [0, 0, 0]

# === Functions ===
def printFinalStats(promptCount, imageGeneratedCount, totalImages, startTime):
    endTime = time.time()
    timeElapsed = endTime - startTime
    hours, remainder = divmod(timeElapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print('')
    logging.info('== Generation complete! ==')
    logging.info('Prompt count: %s', promptCount)
    logging.info('Images generated: %s', imageGeneratedCount)
    logging.info('Total images: %s', totalImages)
    logging.info('Time elapsed: %s', "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def generatePrompts():
    print('')
    logging.info('### Generating prompts... ###')
    promptSets = genPrompt.genPrompts(datasetName=datasetName, configFile='./src/genPrompt/config.json')
    promptSet = promptSets['generic']
    totalPrompts = len(promptSets['generic'])
    totalImages = totalPrompts * 3
    logging.info('total prompts: %s', totalPrompts)

    # save promtSets as json
    with open(os.path.join(outputFolder, datasetName, 'prompts.json'), 'w') as f:
        import json
        json.dump(promptSets, f, indent=4)
    with open(os.path.join(outputFolder, datasetName, 'prompts.txt'), 'w') as f:
        for prompt in promptSet:
            f.write(prompt['prompt'] + '\n')
    logging.info('Prompts written to %s', os.path.join(outputFolder, datasetName, 'prompts.json'))
    logging.info('Prompts written to %s', os.path.join(outputFolder, datasetName, 'prompts.txt'))

    return promptSet, totalPrompts, totalImages

def generateImages():
    # get global variables
    global promptCount
    global imageCount
    global imageGeneratedCount
    global totalImages
    global maxGenerationCount
    global datasetName
    global seed
    global steps

    
    print('')
    logging.info('### Generating images... ###')
    logging.info('Make sure the SD Model is loaded to VRAM (Settings > Actions > Load SD checkpoint to VRAM from RAM)')

    for prompt in promptSet:
        seed = random.randint(1000, 9999999999)
        # === CLEAN ===
        # generate clean image, txt2img
        payloadTxt2img['prompt'] = prompt['prompt'] + ', (clean)'
        payloadTxt2img['seed'] = seed

        # generate
        print('')
        logging.info(f'[{imageCount}/{totalImages}] Generating clean image...')
        logging.info('Prompt: %s', payloadTxt2img['prompt'])
        logging.info('Seed: %s; Steps: %s', seed, steps)

        out_dir = os.path.join(outputFolder, datasetName, 'clean')
        fileName = datasetName + '-clean-' + str(promptCount)
        txt2imgPath, txt2imgResponse = api.call_txt2img_api(out_dir=out_dir, fileName=fileName , **payloadTxt2img)
        logging.info('Image saved to %s', txt2imgPath)
        
        # increment counters
        imageCount += 1
        imageGeneratedCount[0] += 1

        # === avgDirty ===
        # make it slightly dirty, img2img
        promptImg2img = prompt['prompt'] + ', (stains), <lora:dirtyStyle_LoRA_v2-000008:0.35>'
        payloadImg2img['prompt'] = promptImg2img
        payloadImg2img['seed'] = seed

        # load image as init_images
        payloadImg2img['init_images'] = [api.encode_file_to_base64(txt2imgPath)]

        # generate
        print('')
        logging.info(f'[{imageCount}/{totalImages}] Generating slightly dirty image (img2img)...')
        logging.info('Prompt: %s', payloadImg2img['prompt'])
        logging.info('Seed: %s; Steps: %s', seed, steps)

        out_dir = os.path.join(outputFolder, datasetName, 'avgDirty')
        fileName = datasetName + '-avgDirty-' + str(promptCount)
        img2imgPath, img2imgResponse = api.call_img2img_api(out_dir=out_dir, fileName=fileName , **payloadImg2img)
        logging.info('Image saved to %s', img2imgPath)
        
        # increment counters
        imageCount += 1
        imageGeneratedCount[1] += 1

        # === dirty ===
        # generate dirty image, img2img
        promptImg2img = prompt['prompt'] + ', (dirty, stains), <lora:dirtyStyle_LoRA_v2-000008:0.5>'
        payloadImg2img['prompt'] = promptImg2img
        payloadImg2img['seed'] = seed

        # generate
        print('')
        logging.info(f'[{imageCount}/{totalImages}] Generating dirty image (img2img)...')
        logging.info('Prompt: %s', payloadImg2img['prompt'])
        logging.info('Seed: %s; Steps: %s', seed, steps)


        out_dir = os.path.join(outputFolder, datasetName, 'dirty')
        fileName = datasetName + '-dirty-' + str(promptCount)
        img2imgPath, img2imgResponse = api.call_img2img_api(out_dir=out_dir, fileName=fileName , **payloadImg2img)
        logging.info('Image saved to %s', img2imgPath)
        
        # increment counters or break
        if promptCount >= maxGenerationCount:
            break
        else:
            promptCount += 1
        imageCount += 1
        imageGeneratedCount[2] += 1


# ====== Main ======
if __name__ == '__main__':
    startTime = time.time()

    # init APIHandler
    api = sdwebui.APIHandler()
    logging.info('APIHandler initialized')

    try:
        promptSet, totalPrompts, totalImages = generatePrompts()   # get prompts
        generateImages()   # generate images
        
        printFinalStats(promptCount-1, imageGeneratedCount, totalImages, startTime)   # print final stats
        
    except Exception as e:
        logging.error('Error: %s', e)
        logging.error('Error: %s', str(e))
        logging.error('Error: %s', repr(e))
        logging.error('Error: %s', e.args)
        logging.error('Error: %s', e.with_traceback)
        logging.error('Error: %s', e.with_traceback())
        
        printFinalStats(promptCount, imageGeneratedCount, totalImages, startTime)

    except KeyboardInterrupt as e:
        logging.error('KeyboardInterrupt: %s', e)

        logging.info('exiting...')
        printFinalStats(promptCount, imageGeneratedCount, totalImages, startTime)