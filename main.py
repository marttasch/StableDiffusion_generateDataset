import os
import random
import time

import src.A1111_api.A1111_api as sdwebui
import src.genPrompt.genPrompt as genPrompt

import logging

# ===== Config =====
datasetName = 'urinal_v3-testing'

# segmentation
segmentationOutputFolder = 'seg'
safeOutputImagesPerImage = 1   # 3, 1
useOutputImageIndex = 2   # if safeOutputImagesPerImage is 1, select which output image to use (0, 1, 2)

samModelName = 'sam_vit_h_4b8939.pth'


dinoEnabled = True
dinoPrompt = 'toilet'
dinoModelName = 'GroundingDINO_SwinT_OGC (694MB)'   # GroundingDINO_SwinT_OGC (694MB), GroundingDINO_SwinB (938MB)

# Generation
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
def printFinalGenerationStats(promptCount, imageGeneratedCount, totalImages, startTime):
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

    # wait for user to load model to RAM
    userInput = 'n'
    while userInput.lower() != 'y':
        userInput = input('SD Model loaded to RAM? [(y)es/(n)o/(e)nd]: ')
        if userInput.lower() == 'e':
            print('Exiting...')
            exit()

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

def segmentate(classFolder):

    # create output folder
    out_dir = os.path.join(classFolder, segmentationOutputFolder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # get all images in class folder
    images = [os.path.join(classFolder, f) for f in os.listdir(classFolder) if os.path.isfile(os.path.join(classFolder, f))]
    totalImages = len(images)
    logging.info('Segmentating %s images in %s', totalImages, classFolder)

    # segmentate images
    imageCount = 1
    for image in images:
        logging.info(f'Segmentating image [{imageCount}/{totalImages}]: {image}')
        decoded_image = api.encode_file_to_base64(image)
        msg, blended_images, masks, masked_images = api.call_sam_predict_api(decoded_image=decoded_image, samModelName=samModelName, dinoEnabled=dinoEnabled, dinoPrompt=dinoPrompt, dino_model_name=dinoModelName)
        
        # == save images ==
        out_dir = os.path.join(classFolder, segmentationOutputFolder)

        # blended images
        fileName = os.path.basename(image).split('.')[0] + '-blended'
        if safeOutputImagesPerImage == 1:
            save_path = os.path.join(out_dir, (fileName + f'_{useOutputImageIndex}.png'))
            api.decode_and_save_base64(blended_images[useOutputImageIndex], save_path)
        else:    
            for index, img in enumerate(blended_images):
                save_path = os.path.join(out_dir, (fileName + f'_{index}.png'))
                api.decode_and_save_base64(img, save_path)

        # masked images
        fileName = os.path.basename(image).split('.')[0] + '-masked'
        if safeOutputImagesPerImage == 1:
            save_path = os.path.join(out_dir, (fileName + f'_{useOutputImageIndex}.png'))
            api.decode_and_save_base64(masked_images[useOutputImageIndex], save_path)
        else:
            for index, img in enumerate(masked_images):
                save_path = os.path.join(out_dir, (fileName + f'_{index}.png'))
                api.decode_and_save_base64(img, save_path)

        # masks
        fileName = os.path.basename(image).split('.')[0] + '-mask'
        if safeOutputImagesPerImage == 1:
            save_path = os.path.join(out_dir, (fileName + f'_{useOutputImageIndex}.png'))
            api.decode_and_save_base64(masks[useOutputImageIndex], save_path)
        else:
            for index, img in enumerate(masks):
                save_path = os.path.join(out_dir, (fileName + f'_{index}.png'))
                api.decode_and_save_base64(img, save_path)
        
        imageCount += 1

def segmentateImages():
    
    print('')
    logging.info('### Segmentating images... ###')
    logging.info('Make sure the SD Model is loaded to RAM (Settings > Actions > Load SD checkpoint to VRAM from RAM)')

    # wait for user to load model to RAM
    userInput = 'n'
    while userInput.lower() != 'y':
        userInput = input('SD Model loaded to RAM? [(y)es/(n)o/(e)nd]: ')
        if userInput.lower() == 'e':
            print('Exiting...')
            exit()

    # get all images in dataset folder
    datasetFolder = os.path.join(outputFolder, datasetName)
    cleanFolder = os.path.join(datasetFolder, 'clean')
    avgDirtyFolder = os.path.join(datasetFolder, 'avgDirty')
    dirtyFolder = os.path.join(datasetFolder, 'dirty')

    # segmentate images  
    print('')
    logging.info('Segmentate Class: clean (1/3)')
    segmentate(cleanFolder)

    print('')
    logging.info('Segmentate Class: avgDirty (2/3)')
    segmentate(avgDirtyFolder)

    print('')
    logging.info('Segmentate Class: dirty (3/3)')
    segmentate(dirtyFolder)

    
    

# ====== Main ======
if __name__ == '__main__':
    startTime = time.time()

    # init APIHandler
    api = sdwebui.APIHandler()
    logging.info('APIHandler initialized')

    try:
        # generation
        userInput = input('Start generation? (y/n): ')
        if userInput.lower() == 'y':
            promptSet, totalPrompts, totalImages = generatePrompts()   # get prompts
            generateImages()   # generate images
            printFinalGenerationStats(promptCount-1, imageGeneratedCount, totalImages, startTime)   # print final stats

        # segmentation
        userInput = input('Start segmentating? (y/n): ')
        if userInput.lower() == 'y':
            segmentateImages()

        
    except Exception as e:
        logging.error('Error: %s', e)
        logging.error('Error: %s', str(e))
        logging.error('Error: %s', repr(e))
        logging.error('Error: %s', e.args)
        logging.error('Error: %s', e.with_traceback)
        logging.error('Error: %s', e.with_traceback())
        
        printFinalGenerationStats(promptCount, imageGeneratedCount, totalImages, startTime)

    except KeyboardInterrupt as e:
        logging.error('KeyboardInterrupt: %s', e)

        logging.info('exiting...')
        printFinalGenerationStats(promptCount, imageGeneratedCount, totalImages, startTime)