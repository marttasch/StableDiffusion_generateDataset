import json
import sys
import os
from itertools import product

def generate_prompts_from_json(json_data, type="txt2img"):
    prompts = {}
    
    for set in json_data[type]["sets"]:
        set_prompts = []
        set_class = set["dataset_class"]
        for prompt_template in set["prompts"]:
            for combination in product(
                json_data[type]["object_names"],
                json_data[type]["object_materials"],
                json_data[type]["backgrounds"],
                json_data[type]["perspective"],
                json_data[type]["viewpoints"],
                set["dirtiness"],
                set["LoRA"]
            ):
                object_name, object_material, background, perspective, viewpoint, dirtiness, LoRA = combination
                prompt = prompt_template.format(
                    object_name=object_name,
                    object_material=object_material,
                    background=background,
                    perspective=perspective,
                    viewpoint=viewpoint,
                    dirtiness=dirtiness,
                    LoRA=LoRA
                )
                for _ in range(json_data[type]["seeds_per_prompt"]):
                    set_prompts.append({
                        "prompt": prompt,
                        "prompt_template": prompt_template,
                        "object_name": object_name,
                        "object_material": object_material,
                        "background": background,
                        "perspective": perspective,
                        "viewpoint": viewpoint,
                        "dirtiness": dirtiness,
                        "LoRA": LoRA
                    })
        prompts[set_class] = set_prompts
    return prompts

def genPrompts(configFile='config.json'):
    # read config file
    with open(configFile, 'r') as jsonFile:
        promptConfig = json.load(jsonFile)

    # extract promptConfig
    promptsetsName = promptConfig['name']
    # txt2img
    promptSets = generate_prompts_from_json(promptConfig, type="txt2img")

    # create output folder
    outputFolder = 'output/' + promptsetsName + '/'
    print('Output folder:', outputFolder)
    os.makedirs(outputFolder, exist_ok=True)

    # save promtSets as json
    with open('output/' + promptsetsName + '/prompts.json', 'w') as f:
        json.dump(promptSets, f, indent=4)
    print('Prompts written to', outputFolder + 'prompts.json')

    # write prompts set to txt as single file per set, promptsets is a dictionary
    promptCount = 0
    promptTotalList = []
    for set in promptSets:
        promptTotalList.append({'prompt': '== ' + set + ' ==: car'})
        setPrompts = promptSets[set]
        with open(outputFolder + set + '.txt', 'w') as f:
            for prompt in setPrompts:
                f.write(prompt['prompt'] + '\n')
                promptTotalList.append(prompt)
        print('Prompts written to', outputFolder + set + '.txt')
        print('Number of prompts:', len(setPrompts))
        print('Example prompt:', setPrompts[0]['prompt'])
        print('')
        promptCount += len(setPrompts)
        

    with open(outputFolder + 'total.txt', 'w') as f:
        for prompt in promptTotalList:
            f.write(prompt['prompt'] + '\n')
    print('Total number of prompts:', promptCount)

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

    promptList = genPrompts(configFile=configFile)