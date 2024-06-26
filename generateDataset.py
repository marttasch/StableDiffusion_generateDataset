import os
import random
import time
import re
import shutil
import sys

import src.A1111_api as sdwebui
import src.genPrompt as genPrompt
from src.mts_utils import *

import gotifyHandler as gotifyHandler

import logging
import argparse

# ===== Config =====
datasetName = 'urinal_testing'
promptConfigFile = './prompt_config.json'

# generate dataset
loraWeights = [0, 0.4, 0.5]

# prepare dataset
splitRatio = [0.7, 0.2, 0.1]   # train, test, val
filterPrefix = 'original'   # filter for imageType (e.g. 'blended', 'masked', 'binarymask', 'original')

# ----- only change below if necessary -----
datasetOutputFolder = 'datasets'   # folder to save training datasets
outputFolder = 'image_generation_output'   # folder to save generated images

# segmentation
segmentationOutputFolder = 'seg'
safeOutputImagesPerImage = 1   # 3, 1
useOutputImageIndex = 2   # if safeOutputImagesPerImage is 1, select which output image to use (0, 1, 2)

samModelName = 'sam_vit_h_4b8939.pth'

dinoEnabled = True
dinoPrompt = 'toilet'
dinoModelName = 'GroundingDINO_SwinB (938MB)'   # GroundingDINO_SwinT_OGC (694MB), GroundingDINO_SwinB (938MB)

# Generation
maxGenerationCount = 2000   # max number of images to generate

prompt = "puppy dog"   # default prompt, will be overwritten by genPrompt
negative_prompt = "(semi-realistic,cgi,3d,render,sketch,cartoon,drawing,anime),text,cropped,out of frame,cut off,(worst quality,low quality),jpeg artifacts,duplicate,(deformed),blurry,bad proportions,faucet,UnrealisticDream"
seed = -1   # will be overwritten with random seed
steps = 35
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

# Variables
promptCount = 1
imageCount = 1
imageGeneratedCount = [0, 0, 0]

# === Functions ===
def createFolders():
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

def initLogging():
    # check if log file exists, if yes, move to numbered backup like log_1.txt
    loggingPath = os.path.join(outputFolder, datasetName, 'log.txt')
    if os.path.exists(loggingPath):
        i = 1
        while os.path.exists(os.path.join(outputFolder, datasetName, f'log_{i}.txt')):
            i += 1
        shutil.move(loggingPath, os.path.join(outputFolder, datasetName, f'log_{i}.txt'))

    # init logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler(loggingPath, mode='w'),
                            logging.StreamHandler()
                        ])
    
def initSDAPI():
    # init APIHandler
    global api
    api = sdwebui.APIHandler()
    logging.info('APIHandler initialized') 

def printFinalGenerationStats(promptCount, imageGeneratedCount, totalImages, startTime):
    timeElapsedStr = get_TimeElapsed(startTime)
    
    print('')
    logging.info('== Generation complete! ==')
    logging.info('Prompt count: %s', promptCount)
    logging.info('Images generated: %s', imageGeneratedCount)
    logging.info('Total images: %s', totalImages)
    logging.info('Time elapsed: %s', timeElapsedStr)

def printFinalSegmentationStats(startTime):
    timeElapsedStr = get_TimeElapsed(startTime)
    
    print('')
    logging.info('== Segmentation complete! ==')
    logging.info('Time elapsed: %s', timeElapsedStr)


def generatePrompts(configFile=promptConfigFile):
    print('')
    logging.info('### Generating prompts... ###')
    promptSet = genPrompt.genPrompts(datasetName=datasetName, configFile=configFile)
    totalPrompts = len(promptSet)
    totalImages = totalPrompts * 3
    logging.info('total prompts: %s', totalPrompts)

    # save promtSets as json
    with open(os.path.join(outputFolder, datasetName, 'prompts.json'), 'w') as f:
        import json
        json.dump(promptSet, f, indent=4)
    with open(os.path.join(outputFolder, datasetName, 'prompts.txt'), 'w') as f:
        for prompt in promptSet:
            f.write(prompt['prompt'] + '\n')
    # save promptconfig file
    shutil.copyfile(configFile, os.path.join(outputFolder, datasetName, 'prompt_config.json'))

    logging.info('Prompts written to %s', os.path.join(outputFolder, datasetName, 'prompts.json'))
    logging.info('Prompts written to %s', os.path.join(outputFolder, datasetName, 'prompts.txt'))

    return promptSet, totalPrompts, totalImages

def calculateTimeRemaining(imageCount, totalImages, startTime):
    timeElapsed = time.time() - startTime
    timePerImage = timeElapsed / imageCount
    timeRemaining = (totalImages - imageCount) * timePerImage
    timeRemainingStr = get_TimeElapsed(timeRemaining)
    timeRemainingStr = "{:0>2}:{:0>2}:{:05.2f}".format(int(timeRemaining // 3600), int((timeRemaining % 3600) // 60), (timeRemaining % 60))
    timePerImageStr = "{:0>2}:{:05.2f}".format(int((timePerImage % 3600) // 60), (timePerImage % 60))
    logging.info(f'Time per image: {timePerImageStr}; Time remaining: {timeRemainingStr}')
   
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
        userInput = input('SD Model loaded to VRAM? [(y)es/(n)o/(e)nd]: ')
        if userInput.lower() == 'e':
            print('Exiting...')
            exit()

    startTimeperImage = time.time()

    for prompt in promptSet:
        seed = random.randint(1000, 9999999999)
        # === CLEAN ===
        # generate clean image, txt2img
        payloadTxt2img['prompt'] = prompt['prompt'] + ', (clean)'
        payloadTxt2img['seed'] = seed

        # generate
        print('')
        calculateTimeRemaining(imageCount, totalImages, startTimeperImage)
        logging.info(f'[{imageCount}/{totalImages}] Generating clean image...')
        logging.info('Prompt: %s', payloadTxt2img['prompt'])
        logging.info('Seed: %s; Steps: %s', seed, payloadTxt2img['steps'])

        out_dir = os.path.join(outputFolder, datasetName, 'clean')
        fileName = datasetName + '-clean-' + str(promptCount)
        txt2imgPath, txt2imgResponse = api.call_txt2img_api(out_dir=out_dir, fileName=fileName , **payloadTxt2img)
        logging.info('Image saved to %s', txt2imgPath)
        
        # increment counters
        imageCount += 1
        imageGeneratedCount[0] += 1


        # === avgDirty ===
        # make it slightly dirty, img2img
        promptImg2img = prompt['prompt'] + f', (stains), <lora:dirtyStyle_LoRA_v2-000008:{loraWeights[1]}>'
        payloadImg2img['prompt'] = promptImg2img
        payloadImg2img['seed'] = seed

        # load image as init_images
        payloadImg2img['init_images'] = [api.encode_file_to_base64(txt2imgPath)]

        # generate
        print('')
        calculateTimeRemaining(imageCount, totalImages, startTimeperImage)
        logging.info(f'[{imageCount}/{totalImages}] Generating slightly dirty image (img2img)...')
        logging.info('Prompt: %s', payloadImg2img['prompt'])
        logging.info('Seed: %s; Steps: %s', seed, payloadImg2img['steps'])

        out_dir = os.path.join(outputFolder, datasetName, 'avgDirty')
        fileName = datasetName + '-avgDirty-' + str(promptCount)
        img2imgPath, img2imgResponse = api.call_img2img_api(out_dir=out_dir, fileName=fileName , **payloadImg2img)
        logging.info('Image saved to %s', img2imgPath)
        
        # increment counters
        imageCount += 1
        imageGeneratedCount[1] += 1

        # === dirty ===
        # generate dirty image, img2img
        promptImg2img = prompt['prompt'] + f', (dirty, stains), <lora:dirtyStyle_LoRA_v2-000008:{loraWeights[2]}>'
        payloadImg2img['prompt'] = promptImg2img
        payloadImg2img['seed'] = seed

        # generate
        print('')
        calculateTimeRemaining(imageCount, totalImages, startTimeperImage)
        logging.info(f'[{imageCount}/{totalImages}] Generating dirty image (img2img)...')
        logging.info('Prompt: %s', payloadImg2img['prompt'])
        logging.info('Seed: %s; Steps: %s', seed, payloadImg2img['steps'])


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

def segmentate(classFolder, ):

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
    startTime = time.time()
    for image in images:
        logging.info(f'Segmentating image [{imageCount}/{totalImages}]: {image}')
        calculateTimeRemaining(imageCount, totalImages, startTime)
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
        fileName = os.path.basename(image).split('.')[0] + '-binarymask'
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

def createTrainingDataset(folder, datasetName, filterPrefix, outputFolder=None, splitRatio=[0.7, 0.2, 0.1]):
    '''
        Create training dataset from generated/segmentated images

        Args:
            folder (str): folder containing images
            filterPrefix (str): filter for imageType (e.g. 'blended', 'masked', 'binarymask', 'original')
    '''
    logging.info("### Creating training dataset ###")
    logging.info("Folder: %s; FilterPrefix: %s", folder, filterPrefix)

    ignore = []
    if filterPrefix == 'original':
        # ignore 'seg' subfolder
        ignore = ['seg']

    # get classes from folder names inside the folder, then get all images from each class including subfolders
    classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    logging.info('Classes: %s', classes)

    images = {}
    # crawl through each class folder and get all images per class, including subfolders
    for c in classes:
        #images[c] = [os.path.join(folder, c, f) for f in os.listdir(os.path.join(folder, c)) if os.path.isfile(os.path.join(folder, c, f))]
        images[c] = []
        for root, dirs, files in os.walk(os.path.join(folder, c)):
            # ignore subfolders
            for i in ignore:
                if i in dirs:
                    dirs.remove(i)
            # get images
            for file in files:
                # filter for filterprefix and image file extensions
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    if (filterPrefix == 'original') or (filterPrefix in file):
                        images[c].append(os.path.join(root, file))

        # sort images by filename, 
        images[c].sort(key=lambda f: int(re.sub(r'\D', '', f)))

    # split images into train, test, val
    trainImages = {}
    testImages = {}
    valImages = {}
    for c in classes:
        totalImages = len(images[c])
        trainCount = int(totalImages * splitRatio[0])
        testCount = int(totalImages * splitRatio[1])
        valCount = totalImages - trainCount - testCount

        trainImages[c] = images[c][:trainCount]
        testImages[c] = images[c][trainCount:trainCount+testCount]
        valImages[c] = images[c][trainCount+testCount:]

    # print table
    print(f'{"class":<10} | {"train":<5} | {"test":<5} | {"val":<5}')
    for c in trainImages:
        print(f'{c:<10} | {len(trainImages[c]):<5} | {len(testImages[c]):<5} | {len(valImages[c]):<5}')
    print('')

    # move images to train, test, val folders
    # create folder structure
    if outputFolder is None:
        outputFolder = os.path.join(os.getcwd(), datasetOutputFolder)
    outputFolder = os.path.join(outputFolder, datasetName)
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    for folder in ['train', 'test', 'val']:
        if not os.path.exists(os.path.join(outputFolder, folder)):
            os.makedirs(os.path.join(outputFolder, folder))
        for c in classes:
            if not os.path.exists(os.path.join(outputFolder, folder, c)):
                os.makedirs(os.path.join(outputFolder, folder, c))

    # copy images
    logging.info(f'Copy images to {outputFolder}...')
    for c in trainImages:
        for img in trainImages[c]:
            imgName = os.path.basename(img)
            imgPath = os.path.join(outputFolder, 'train', c, imgName)
            shutil.copy(img, imgPath)
    for c in testImages:
        for img in testImages[c]:
            imgName = os.path.basename(img)
            imgPath = os.path.join(outputFolder, 'test', c, imgName)
            shutil.copy(img, imgPath)
    for c in valImages:
        for img in valImages[c]:
            imgName = os.path.basename(img)
            imgPath = os.path.join(outputFolder, 'val', c, imgName)
            shutil.copy(img, imgPath)

def sendGotifyMessage(title, message, priority=7):
    gotifyHandler.send_message(title, message, priority)
      

# ====== Main ======
if __name__ == '__main__':
    startTime = time.time()

    # --- parse arguments ---
    parser = argparse.ArgumentParser(description= '''
                                        Generate prompts using prompt-templates, generate datasets using A1111 SDWebUI API 
                                        and segmentate images using A1111 SAM API.
                                    ''')
    parser.add_argument('--datasetName', type=str, help='Name of the dataset')
    parser.add_argument('--promptConfigFile', type=str, help='Path to the prompt config file')
    parser.add_argument('--generateDataset', type=str, help='Generate dataset (True/False)')
    parser.add_argument('--segmentateImages', type=str, help='Segmentate images (True/False)')
    parser.add_argument('--createTrainingDataset', type=str, help='Create training dataset (True/False)')

    args = parser.parse_args()
    if all(value is None for value in vars(args).values()):
        print('No arguments given. Using default values.')
        print(parser.print_help())
    if args.datasetName:
        datasetName = args.datasetName
        print(f'Using dataset name from cmd: {datasetName}')
    if args.promptConfigFile:
        promptConfigFile = args.promptConfigFile
        print(f'Using prompt config file from cmd: {promptConfigFile}')
    
    print(f'Arguments: {args}')
    print(f'DatasetName: {datasetName}')
    print(f'PromptConfigFile: {promptConfigFile}')
    print(f'GenerateDataset: {args.generateDataset}')
    print(f'SegmentateImages: {args.segmentateImages}')
    print(f'CreateTrainingDataset: {args.createTrainingDataset}')

    if args.generateDataset == "True":
        print('ARGUMENTS: generateDataset=True')


    # --- start ---
    print(f'\n=== Dataset Generation ===')
    # ask user if datasename is correct, if not, change it
    userInput = input(f'Use dataset name "{datasetName}"? (y/n): ')
    if userInput.lower() != 'y':
        datasetName = input('Enter dataset name: ')
    # create output folders
    createFolders()

    # ask user if promptConfigFile is correct, if not, change it
    userInput = input(f'Use prompt config file "{promptConfigFile}"? (y/n): ')
    if userInput.lower() != 'y':
        promptConfigFile = input('Enter prompt config file: ')
    # check if promptConfigFile exists
    if not os.path.exists(promptConfigFile):
        print(f'Error: promptConfigFile "{promptConfigFile}" not found!')
        exit()
    
    # init
    initLogging()
    initSDAPI()
    
    try:
        # generation
        userInput = 'n'
        if not args.generateDataset:
            userInput = input('Start generation? (y/n): ')
        if userInput.lower() == 'y' or args.generateDataset == "True":
            startTime = time.time()
            promptSet, totalPrompts, totalImages = generatePrompts(configFile=promptConfigFile)  # get prompts
            generateImages()   # generate images
            printFinalGenerationStats(promptCount-1, imageGeneratedCount, totalImages, startTime)   # print final stats

            # send message
            timeElapsed = get_TimeElapsed(startTime)
            sendGotifyMessage(
                title=f'Dataset-Generation complete ({datasetName})',
                message=f'Prompt count: {promptCount-1}\nImages generated: {imageGeneratedCount}\nTotal images: {totalImages}\nTime elapsed: {timeElapsed}'
            )

        # segmentation
        userInput = 'n'
        if not args.segmentateImages:
            userInput = input('Start segmentating? (y/n): ')
        if userInput.lower() == 'y' or args.segmentateImages == "True":
            startTime = time.time()
            segmentateImages()
            printFinalSegmentationStats(startTime)   # print final stats

            # send message
            timeElapsed = get_TimeElapsed(startTime)
            sendGotifyMessage(
                title=f'Dataset-Segmentation complete ({datasetName})',
                message=f'Images segmentated in {timeElapsed}'
            )

        # create training dataset
        userInput = 'n'
        if not args.createTrainingDataset:
            userInput = input('Create training dataset? (y/n): ')
        if userInput.lower() == 'y' or args.createTrainingDataset == "True":
            startTime = time.time()
            createTrainingDataset(os.path.join(outputFolder, datasetName), datasetName, filterPrefix=filterPrefix, outputFolder=None, splitRatio=splitRatio)
            logging.info('Training dataset created')

            # send message
            timeElapsed = get_TimeElapsed(startTime)
            sendGotifyMessage(
                title=f'Training-Dataset created ({datasetName})',
                message=f'Training dataset created in {timeElapsed}'
            )
        
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