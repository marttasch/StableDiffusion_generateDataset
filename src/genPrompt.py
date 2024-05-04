import json
import sys
import os
from itertools import product

def generate_prompts_from_json(json_data, type="txt2img"):
    prompts = []
    
    for prompt_template in json_data[type]["prompts"]:
        for combination in product(
            json_data[type]["object_names"],
            json_data[type]["object_materials"],
            json_data[type]["backgrounds"],
            json_data[type]["perspective"],
            json_data[type]["viewpoints"]
        ):
            object_name, object_material, background, perspective, viewpoint = combination
            prompt = prompt_template.format(
                object_name=object_name,
                object_material=object_material,
                background=background,
                perspective=perspective,
                viewpoint=viewpoint
            )
            for _ in range(json_data[type]["seeds_per_prompt"]):
                prompts.append({
                    "prompt": prompt,
                    "prompt_template": prompt_template,
                    "object_name": object_name,
                    "object_material": object_material,
                    "background": background,
                    "perspective": perspective,
                    "viewpoint": viewpoint
                })
    return prompts

def genPrompts(datasetName=None, configFile='config.json', saveAsFile=False):
    # read config file
    with open(configFile, 'r') as jsonFile:
        promptConfig = json.load(jsonFile)

    # extract promptConfig
    promptsetsName = promptConfig['name']
    if datasetName:
        promptsetsName = datasetName

    # get prompts
    promptSets = generate_prompts_from_json(promptConfig, type="txt2img")

    if saveAsFile:
        # save prompts to file
        promptFileJson = os.path.join('prompts', f'{promptsetsName}.json')
        promptFileTxt = os.path.join('prompts', f'{promptsetsName}.txt')
        if not os.path.exists('prompts'):
            os.makedirs('prompts')
        with open(promptFileJson, 'w') as jsonFile:
            json.dump(promptSets, jsonFile, indent=4)
        with open(promptFileTxt, 'w') as txtFile:
            for prompt in promptSets:
                txtFile.write(prompt['prompt'] + '\n')
            print(f"Prompts saved to {promptFileJson}")
            print(f"Prompts saved to {promptFileTxt}")
    return promptSets


if __name__ == '__main__':
    print("\n### Generate prompts ###")
    # get arguments from command line
    if len(sys.argv) < 2:
        print("Missing argument: configFile")
        print("Usage: `python genPrompt.py <configFile>`")
        print("Will use default `config.json`\n")
        configFile = 'config.json'
    else:
        configFile = sys.argv[1]
        print("Using config file: ", configFile) 

    promptList = genPrompts(configFile=configFile, saveAsFile=True)